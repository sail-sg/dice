import re
import pandas as pd

def extract_numbers(s):
    # Pattern to match positive and negative integers, and numbers with commas
    pattern = r"-?\d+[\d,]*"
    # Remove commas to handle numbers like '65,000'
    numbers = [re.sub(",", "", match) for match in re.findall(pattern, s)]
    if numbers:
        return numbers[0]
    else:
        print(s)
        return None
    
def exact_match_numbers(
    generated: list,
    reference: list,
): 
    data = pd.DataFrame(
        {"generated": generated, "reference": reference}
    )
    data["generated"] = data["generated"].apply(
        lambda x: extract_numbers(x) if x else None
    )
    data["reference"] = data["reference"].apply(
        lambda x: extract_numbers(x) if x else None
    )

    # use metrics
    acc = (data["reference"] == data["generated"]).sum() / len(data)
    return {"accuracy": acc}
