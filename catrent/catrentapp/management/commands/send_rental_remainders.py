from django.core.management.base import BaseCommand
from django.utils import timezone
from django.core.mail import send_mail
from catrentapp.models import Rental, Operator
from django.conf import settings
from datetime import timedelta

# For testing
REMINDER_DAYS = [4,3,2,1,0]  # Today

class Command(BaseCommand):
    help = "Send email reminders to operators for upcoming rental check-ins (no logging)"

    def handle(self, *args, **kwargs):
        today = timezone.now().date()
        print(f"[DEBUG] Today is {today}")

        for days_before in REMINDER_DAYS:
            reminder_date = today + timedelta(days=days_before)
            print(f"[DEBUG] Checking rentals for reminder in {days_before} days (reminder_date={reminder_date})")

            rentals_due = Rental.objects.filter(
                active=True,
                expected_end_date__date=reminder_date
            )

            if not rentals_due.exists():
                print(f"[DEBUG] No rentals due on {reminder_date}")
                continue

            for rental in rentals_due:
                print(f"[DEBUG] Found rental {rental.rental_id}, machine {rental.machine.equipment_id}")

                # Get operator email
                operator = Operator.objects.filter(operator_id=rental.operator_id).first()
                if not operator or not operator.email:
                    print(f"[DEBUG] No email found for rental {rental.rental_id}")
                    continue

                operator_email = operator.email

                # Email content
                if days_before == 0:
                    subject_prefix = "Today: "
                    message_intro = "This is a reminder that your rental is due today"
                else:
                    subject_prefix = f"In {days_before} days: "
                    message_intro = f"This is a reminder that your rental is due in {days_before} days"

                subject = f"{subject_prefix}Rental {rental.rental_id} check-in"
                message = f"""
Hello {rental.operator_name},

{message_intro} for machine {rental.machine.equipment_id} ({rental.machine.type}) 
scheduled for check-in on {rental.expected_end_date.strftime('%Y-%m-%d %H:%M')}.

Please ensure the equipment is returned on time.

Thank you,
CatRent Team
"""

                print(f"[DEBUG] Sending email to {operator_email}...")
                send_mail(
                    subject,
                    message,
                    settings.DEFAULT_FROM_EMAIL,
                    [operator_email],
                    fail_silently=False,
                )

                print(f"[DEBUG] Reminder sent for rental {rental.rental_id}")
