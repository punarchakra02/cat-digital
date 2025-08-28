import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging
from datetime import datetime, timedelta
import json
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('anomaly_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AnomalyDetector:
    def __init__(self, n_neighbors=5, contamination=0.1, n_clusters=3):
        """
        Initialize the KNN-based anomaly detector with clustering for anomaly classification
        
        Parameters:
        n_neighbors (int): Number of neighbors to consider
        contamination (float): Expected proportion of anomalies in the data
        n_clusters (int): Number of clusters for anomaly classification
        """
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.n_clusters = n_clusters
        self.knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto')
        self.scaler = StandardScaler()
        self.threshold = None
        self.is_fitted = False
        self.anomaly_clusters = None
        self.cluster_centers = None
        self.feature_names = None
        self.normal_stats = None
        
    def fit(self, X, feature_names=None):
        """
        Fit the anomaly detector on the training data
        
        Parameters:
        X (array-like): Training data of shape (n_samples, n_features)
        feature_names (list): Names of the features
        """
        self.feature_names = feature_names
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Calculate normal statistics for each feature
        self.normal_stats = {
            'mean': np.mean(X_scaled, axis=0),
            'std': np.std(X_scaled, axis=0),
            'min': np.min(X_scaled, axis=0),
            'max': np.max(X_scaled, axis=0)
        }
        
        # Fit the KNN model
        self.knn.fit(X_scaled)
        
        # Calculate distances to k-th nearest neighbor for all training points
        distances, _ = self.knn.kneighbors(X_scaled)
        kth_distances = distances[:, -1]  # Distance to k-th nearest neighbor
        
        # Set threshold based on contamination level
        self.threshold = np.percentile(kth_distances, (1 - self.contamination) * 100)
        
        # Identify anomalies in training data for clustering
        training_anomalies = kth_distances > self.threshold
        if np.sum(training_anomalies) > self.n_clusters:
            # Cluster anomalies to identify different types
            anomaly_data = X_scaled[training_anomalies]
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
            self.anomaly_clusters = kmeans.fit_predict(anomaly_data)
            self.cluster_centers = kmeans.cluster_centers_
        else:
            self.anomaly_clusters = np.zeros(np.sum(training_anomalies))
            self.cluster_centers = None
            
        self.is_fitted = True
        
        logger.info(f"Model fitted successfully!")
        logger.info(f"Threshold set at: {self.threshold:.4f}")
        logger.info(f"Expected anomaly rate: {self.contamination:.1%}")
        logger.info(f"Number of anomaly clusters: {self.n_clusters}")
        
        return self
    
    def classify_anomaly(self, X_scaled, distance, index):
        """
        Classify the type of anomaly based on feature deviations and clustering
        
        Parameters:
        X_scaled (array): Scaled feature values
        distance (float): Distance to k-th nearest neighbor
        index (int): Index of the data point
        
        Returns:
        dict: Anomaly classification details
        """
        if self.feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(X_scaled.shape[1])]
        else:
            feature_names = self.feature_names
            
        # Calculate feature deviations from normal
        deviations = X_scaled[0] - self.normal_stats['mean']
        deviation_percentages = (deviations / self.normal_stats['std']) * 100
        
        # Find most deviant features
        abs_deviations = np.abs(deviation_percentages)
        top_deviant_indices = np.argsort(abs_deviations)[-3:]  # Top 3 most deviant
        
        # Determine anomaly type based on deviations
        anomaly_type = self._determine_anomaly_type(deviations, deviation_percentages, feature_names)
        
        # Get cluster assignment if clustering was performed
        cluster_id = None
        if self.cluster_centers is not None:
            cluster_distances = np.linalg.norm(self.cluster_centers - X_scaled[0], axis=1)
            cluster_id = np.argmin(cluster_distances)
        
        return {
            'index': index,
            'anomaly_score': distance,
            'anomaly_type': anomaly_type,
            'cluster_id': cluster_id,
            'severity': self._calculate_severity(distance),
            'top_deviant_features': [
                {
                    'feature': feature_names[i],
                    'deviation': deviations[i],
                    'deviation_percentage': deviation_percentages[i]
                }
                for i in top_deviant_indices
            ],
            'all_deviations': dict(zip(feature_names, deviation_percentages))
        }
    
    def _determine_anomaly_type(self, deviations, deviation_percentages, feature_names):
        """
        Determine the type of anomaly based on feature deviations
        """
        # Count positive and negative deviations
        positive_deviations = np.sum(deviation_percentages > 20)  # 20% threshold
        negative_deviations = np.sum(deviation_percentages < -20)
        
        # Find the most extreme deviation
        max_deviation_idx = np.argmax(np.abs(deviation_percentages))
        max_deviation = deviation_percentages[max_deviation_idx]
        max_feature = feature_names[max_deviation_idx]
        
        # Classify based on patterns
        if positive_deviations > negative_deviations and positive_deviations >= 2:
            return f"High-Value Anomaly (Multiple features elevated, max: {max_feature} +{max_deviation:.1f}%)"
        elif negative_deviations > positive_deviations and negative_deviations >= 2:
            return f"Low-Value Anomaly (Multiple features depressed, max: {max_feature} {max_deviation:.1f}%)"
        elif abs(max_deviation) > 50:
            return f"Extreme Deviation Anomaly ({max_feature}: {max_deviation:.1f}%)"
        elif np.sum(np.abs(deviation_percentages) > 30) >= 3:
            return f"Multi-Feature Anomaly ({np.sum(np.abs(deviation_percentages) > 30)} features affected)"
        else:
            return f"General Anomaly (max deviation: {max_feature} {max_deviation:.1f}%)"
    
    def _calculate_severity(self, distance):
        """
        Calculate anomaly severity based on distance
        """
        if distance > self.threshold * 2:
            return "CRITICAL"
        elif distance > self.threshold * 1.5:
            return "HIGH"
        elif distance > self.threshold * 1.2:
            return "MEDIUM"
        else:
            return "LOW"
    
    def predict(self, X):
        """
        Predict anomalies in the data
        
        Parameters:
        X (array-like): Data to predict anomalies for
        
        Returns:
        array: 1 for anomalies, 0 for normal points
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_scaled = self.scaler.transform(X)
        distances, _ = self.knn.kneighbors(X_scaled)
        kth_distances = distances[:, -1]
        
        # Points with distance > threshold are considered anomalies
        predictions = (kth_distances > self.threshold).astype(int)
        
        return predictions
    
    def predict_proba(self, X):
        """
        Get anomaly scores (distance to k-th nearest neighbor)
        
        Parameters:
        X (array-like): Data to get scores for
        
        Returns:
        array: Anomaly scores (higher = more anomalous)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_scaled = self.scaler.transform(X)
        distances, _ = self.knn.kneighbors(X_scaled)
        kth_distances = distances[:, -1]
        
        return kth_distances
    
    def detect_and_log_anomalies(self, X, timestamps=None):
        """
        Detect anomalies and provide detailed logging with classification
        
        Parameters:
        X (array-like): Data to analyze
        timestamps (array-like): Timestamps for each data point
        
        Returns:
        tuple: (predictions, scores, anomaly_details)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_scaled = self.scaler.transform(X)
        distances, _ = self.knn.kneighbors(X_scaled)
        kth_distances = distances[:, -1]
        
        # Points with distance > threshold are considered anomalies
        predictions = (kth_distances > self.threshold).astype(int)
        
        # Detailed anomaly analysis
        anomaly_details = []
        anomaly_count = 0
        
        for i, (pred, distance) in enumerate(zip(predictions, kth_distances)):
            if pred == 1:  # Anomaly detected
                anomaly_count += 1
                anomaly_info = self.classify_anomaly(X_scaled[i:i+1], distance, i)
                
                # Add timestamp if available
                if timestamps is not None and i < len(timestamps):
                    try:
                        anomaly_info['timestamp'] = timestamps.iloc[i] if hasattr(timestamps, 'iloc') else timestamps[i]
                    except (IndexError, KeyError):
                        anomaly_info['timestamp'] = f"Index_{i}"
                
                anomaly_details.append(anomaly_info)
                
                # Log detailed anomaly information
                self._log_anomaly(anomaly_info, anomaly_count)
        
        logger.info(f"=== Anomaly Detection Summary ===")
        logger.info(f"Total samples analyzed: {len(predictions)}")
        logger.info(f"Anomalies detected: {anomaly_count}")
        logger.info(f"Anomaly rate: {anomaly_count/len(predictions):.2%}")
        
        return predictions, kth_distances, anomaly_details
    
    def _log_anomaly(self, anomaly_info, anomaly_number):
        """
        Log detailed information about a detected anomaly
        """
        timestamp_str = f" at {anomaly_info.get('timestamp', 'Unknown time')}" if 'timestamp' in anomaly_info else ""
        
        logger.warning(f"🚨 ANOMALY #{anomaly_number} DETECTED{timestamp_str}")
        logger.warning(f"   Index: {anomaly_info['index']}")
        logger.warning(f"   Type: {anomaly_info['anomaly_type']}")
        logger.warning(f"   Severity: {anomaly_info['severity']}")
        logger.warning(f"   Anomaly Score: {anomaly_info['anomaly_score']:.4f}")
        
        if anomaly_info['cluster_id'] is not None:
            logger.warning(f"   Cluster ID: {anomaly_info['cluster_id']}")
        
        logger.warning(f"   Top Deviant Features:")
        for feature_info in anomaly_info['top_deviant_features']:
            direction = "+" if feature_info['deviation_percentage'] > 0 else ""
            logger.warning(f"     • {feature_info['feature']}: {direction}{feature_info['deviation_percentage']:.1f}%")
        
        logger.warning("")  # Empty line for readability
    
    def evaluate(self, X, y_true=None, timestamps=None):
        """
        Evaluate the model and provide insights
        
        Parameters:
        X (array-like): Data to evaluate
        y_true (array-like, optional): True labels if available
        timestamps (array-like, optional): Timestamps for each data point
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        # Get predictions, scores, and detailed anomaly information
        predictions, scores, anomaly_details = self.detect_and_log_anomalies(X, timestamps)
        
        # Calculate statistics
        n_anomalies = np.sum(predictions)
        n_total = len(predictions)
        anomaly_rate = n_anomalies / n_total
        
        logger.info(f"\n=== Model Evaluation ===")
        logger.info(f"Total samples: {n_total}")
        logger.info(f"Detected anomalies: {n_anomalies}")
        logger.info(f"Anomaly rate: {anomaly_rate:.2%}")
        logger.info(f"Score statistics:")
        logger.info(f"  - Mean: {np.mean(scores):.4f}")
        logger.info(f"  - Std: {np.std(scores):.4f}")
        logger.info(f"  - Min: {np.min(scores):.4f}")
        logger.info(f"  - Max: {np.max(scores):.4f}")
        
        if y_true is not None:
            from sklearn.metrics import classification_report, confusion_matrix
            logger.info(f"\n=== Classification Report ===")
            logger.info(f"\n{classification_report(y_true, predictions)}")
            
            logger.info(f"\n=== Confusion Matrix ===")
            cm = confusion_matrix(y_true, predictions)
            logger.info(f"\n{cm}")
        
        return predictions, scores, anomaly_details

def load_and_preprocess_data(file_path='sensor_data_5min.csv', sample_size=None):
    """
    Load and preprocess sensor data from CSV file
    
    Parameters:
    file_path (str): Path to the CSV file
    sample_size (int): Number of samples to use (None for all data)
    
    Returns:
    tuple: (features, labels if available, feature_names, timestamps)
    """
    try:
        # Load the data
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully from {file_path}")
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        
        # Sample data if requested (for faster processing)
        if sample_size is not None and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
            logger.info(f"Sampled {sample_size} rows for faster processing")
            logger.info(f"New shape: {df.shape}")
        
        logger.info(f"\nFirst few rows:")
        logger.info(f"\n{df.head()}")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            logger.info(f"\nMissing values found:")
            logger.info(f"{missing_values[missing_values > 0]}")
            # Fill missing values with forward fill then backward fill
            df = df.ffill().bfill()
            logger.info("Missing values filled using forward/backward fill")
        
        # Extract timestamps if available
        timestamps = None
        timestamp_columns = ['timestamp', 'time', 'datetime', 'date']
        for col in timestamp_columns:
            if col in df.columns:
                try:
                    timestamps = pd.to_datetime(df[col])
                    logger.info(f"Found timestamp column: {col}")
                    break
                except:
                    continue
        
        # Separate features and labels (if 'anomaly' or 'label' column exists)
        label_columns = ['anomaly', 'label', 'target', 'class']
        
        # Get only numeric columns for features (exclude text/datetime columns)
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_columns = [col for col in numeric_columns if col.lower() not in label_columns]
        
        # Remove 'sno' column if it exists (as it's just an index)
        if 'sno' in feature_columns:
            feature_columns.remove('sno')
        
        logger.info(f"Selected numeric feature columns: {feature_columns}")
        
        features = df[feature_columns].values
        labels = None
        
        for label_col in label_columns:
            if label_col in df.columns:
                labels = df[label_col].values
                logger.info(f"Found label column: {label_col}")
                break
        
        logger.info(f"\nFeatures shape: {features.shape}")
        if labels is not None:
            logger.info(f"Labels shape: {labels.shape}")
            logger.info(f"Label distribution: {np.bincount(labels.astype(int))}")
        
        return features, labels, feature_columns, timestamps
        
    except FileNotFoundError:
        logger.error(f"Error: File {file_path} not found!")
        return None, None, None, None
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None, None, None, None

def save_anomaly_report(anomaly_details, output_file='anomaly_report.json'):
    """
    Save detailed anomaly report to JSON file
    
    Parameters:
    anomaly_details (list): List of anomaly information dictionaries
    output_file (str): Output file path
    """
    # Convert numpy arrays to lists for JSON serialization
    serializable_details = []
    for detail in anomaly_details:
        serializable_detail = detail.copy()
        # Convert numpy arrays to lists
        for key, value in serializable_detail.items():
            if isinstance(value, np.ndarray):
                serializable_detail[key] = value.tolist()
            elif isinstance(value, np.integer):
                serializable_detail[key] = int(value)
            elif isinstance(value, np.floating):
                serializable_detail[key] = float(value)
        serializable_details.append(serializable_detail)
    
    with open(output_file, 'w') as f:
        json.dump(serializable_details, f, indent=2, default=str)
    
    logger.info(f"Detailed anomaly report saved to {output_file}")

def visualize_results(X, predictions, scores, anomaly_details, feature_names=None, n_features_to_plot=3):
    """
    Visualize the anomaly detection results
    
    Parameters:
    X (array-like): Original data
    predictions (array): Anomaly predictions
    scores (array): Anomaly scores
    anomaly_details (list): Detailed anomaly information
    feature_names (list): Names of features
    n_features_to_plot (int): Number of features to plot
    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Anomaly scores distribution
    axes[0, 0].hist(scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(np.mean(scores), color='red', linestyle='--', label=f'Mean: {np.mean(scores):.3f}')
    axes[0, 0].axvline(np.percentile(scores, 95), color='orange', linestyle='--', 
                       label=f'95th percentile: {np.percentile(scores, 95):.3f}')
    axes[0, 0].set_title('Distribution of Anomaly Scores')
    axes[0, 0].set_xlabel('Anomaly Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Anomaly vs Normal points (first 2 features)
    normal_mask = predictions == 0
    anomaly_mask = predictions == 1
    
    if X.shape[1] >= 2:
        axes[0, 1].scatter(X[normal_mask, 0], X[normal_mask, 1], 
                          c='blue', alpha=0.6, label='Normal', s=20)
        axes[0, 1].scatter(X[anomaly_mask, 0], X[anomaly_mask, 1], 
                          c='red', alpha=0.8, label='Anomaly', s=30)
        axes[0, 1].set_xlabel(f'Feature 1 ({feature_names[0] if feature_names else "X1"})')
        axes[0, 1].set_ylabel(f'Feature 2 ({feature_names[1] if feature_names else "X2"})')
        axes[0, 1].set_title('Anomaly Detection Results (First 2 Features)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Time series of anomaly scores (if data is time-series)
    axes[1, 0].plot(scores, alpha=0.7, color='green')
    axes[1, 0].scatter(np.where(predictions == 1)[0], 
                      scores[predictions == 1], 
                      c='red', s=30, alpha=0.8, label='Anomalies')
    axes[1, 0].set_title('Anomaly Scores Over Time')
    axes[1, 0].set_xlabel('Sample Index')
    axes[1, 0].set_ylabel('Anomaly Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Anomaly types distribution
    if anomaly_details:
        anomaly_types = [detail['anomaly_type'].split('(')[0].strip() for detail in anomaly_details]
        type_counts = pd.Series(anomaly_types).value_counts()
        
        bars = axes[1, 1].bar(range(len(type_counts)), type_counts.values, 
                             color=['red', 'orange', 'yellow', 'green'][:len(type_counts)])
        axes[1, 1].set_title('Distribution of Anomaly Types')
        axes[1, 1].set_xlabel('Anomaly Types')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_xticks(range(len(type_counts)))
        axes[1, 1].set_xticklabels(type_counts.index, rotation=45, ha='right')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add count values on bars
        for bar, count in zip(bars, type_counts.values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{count}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to run the anomaly detection pipeline
    """
    logger.info("=== KNN Anomaly Detection System with Detailed Classification ===\n")
    
    # File path - update this to your actual file path
    file_path = "sensor_data_5min.csv"
    
    # Load and preprocess data
    # Using sample_size=10000 for faster processing - remove or increase for full dataset
    features, labels, feature_names, timestamps = load_and_preprocess_data(file_path, sample_size=10000)
    
    if features is None:
        logger.error("Could not load data. Please check the file path.")
        return
    
    # Split data if labels are available
    if labels is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        logger.info(f"\nTraining set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        
        # Split timestamps accordingly
        if timestamps is not None:
            # Reset index to ensure proper indexing after sampling
            timestamps_reset = timestamps.reset_index(drop=True)
            train_timestamps = timestamps_reset[:len(X_train)]
            test_timestamps = timestamps_reset[len(X_train):len(X_train)+len(X_test)]
        else:
            train_timestamps = None
            test_timestamps = None
    else:
        # If no labels, use all data for training
        X_train = features
        X_test = features
        y_train = None
        y_test = None
        if timestamps is not None:
            # Reset index to ensure proper indexing after sampling
            timestamps_reset = timestamps.reset_index(drop=True)
            train_timestamps = timestamps_reset
            test_timestamps = timestamps_reset
        else:
            train_timestamps = None
            test_timestamps = None
        logger.info(f"\nUsing all {X_train.shape[0]} samples for training")
    
    # Initialize and train the anomaly detector
    logger.info("\n=== Training Anomaly Detector ===")
    detector = AnomalyDetector(n_neighbors=5, contamination=0.1, n_clusters=3)
    detector.fit(X_train, feature_names)
    
    # Make predictions with detailed analysis
    logger.info("\n=== Making Predictions with Detailed Analysis ===")
    predictions, scores, anomaly_details = detector.evaluate(X_test, y_test, test_timestamps)
    
    # Save detailed anomaly report
    save_anomaly_report(anomaly_details, 'detailed_anomaly_report.json')
    
    # Visualize results
    logger.info("\n=== Generating Visualizations ===")
    visualize_results(X_test, predictions, scores, anomaly_details, feature_names)
    
    # Save results
    results_df = pd.DataFrame({
        'anomaly_score': scores,
        'predicted_anomaly': predictions
    })
    
    if y_test is not None:
        results_df['true_label'] = y_test
    
    if test_timestamps is not None:
        results_df['timestamp'] = test_timestamps
    
    results_df.to_csv('anomaly_detection_results.csv', index=False)
    logger.info(f"\nResults saved to 'anomaly_detection_results.csv'")
    
    # Print summary
    logger.info(f"\n=== Summary ===")
    logger.info(f"Total samples analyzed: {len(predictions)}")
    logger.info(f"Anomalies detected: {np.sum(predictions)}")
    logger.info(f"Anomaly rate: {np.sum(predictions)/len(predictions):.2%}")
    
    if y_test is not None:
        accuracy = np.mean(predictions == y_test)
        logger.info(f"Accuracy: {accuracy:.2%}")
    
    # Print anomaly type summary
    if anomaly_details:
        logger.info(f"\n=== Anomaly Type Summary ===")
        anomaly_types = [detail['anomaly_type'].split('(')[0].strip() for detail in anomaly_details]
        type_counts = pd.Series(anomaly_types).value_counts()
        for anomaly_type, count in type_counts.items():
            logger.info(f"{anomaly_type}: {count} occurrences")

if __name__ == "__main__":
    main()
