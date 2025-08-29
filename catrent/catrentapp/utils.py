def generate_operator_id():
    from .models import Operator
    last_operator = Operator.objects.order_by('-id').first()
    if last_operator:
        last_id = int(last_operator.operator_id.replace("OPR", ""))
        return f"OPR{last_id+1:03d}"
    return "OPR001"