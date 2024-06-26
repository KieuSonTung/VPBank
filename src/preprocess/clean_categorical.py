def convert_edu(status):
    status_stripped = str(status).strip()
    if status_stripped in ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th', 'HS-grad']:
        return 'undergrad'
    elif status_stripped in ['Some-college', 'Bachelors', 'Masters', 'Assoc-voc', 'Assoc-acdm', 'Prof-school', 'Doctorate']:
        return 'grad'

def convert_marital_status(status):
    status_stripped = status.strip()

    if status_stripped in ['Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse']:
        return 'married'
    elif status_stripped in ['Never-married', 'Separated', 'Widowed']:
        return 'single'
    elif status_stripped == 'Divorced':
        return 'divorced'
    else:
        return 'unknown'