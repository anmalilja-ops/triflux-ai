import math

def calculate_with_validation():
    print("=== 3-POINT VALIDATION AI CALCULATOR ===")
    
    # 1. Gather the two main baseline points (A and B)
    print("\n--- Step 1: Input Baseline Data ---")
    e1 = float(input("Enter Baseline Epoch 1 (e.g., 100): "))
    v1 = float(input("Enter Test Accuracy at Baseline Epoch 1 (%): "))
    
    e2 = float(input("Enter Peak Epoch 2 (e.g., 200): "))
    v2 = float(input("Enter Test Accuracy at Peak Epoch 2 (%): "))
    
    # 2. Gather the validation point (C) and the future target (D)
    print("\n--- Step 2: Input Validation & Target Goals ---")
    e_val = float(input("Enter Validation Epoch (between E1 and E2, e.g., 150): "))
    v_val = float(input("Enter ACTUAL Accuracy at Validation Epoch (%): "))
    
    e_future = float(input("Enter the Future Epoch you want to predict (e.g., 250): "))
    
    # 3. Calculate the base progress bit from E1 to E2
    doublings_between = math.log2(e2 / e1)
    actual_gap = v2 - v1
    
    if doublings_between > 0:
        base_bit = actual_gap / (2 * (1 - (0.5 ** doublings_between)))
    else:
        base_bit = actual_gap
        
    print(f"\n[Calculated] Base Progress Bit: +{base_bit:.5f}%")
    
    # 4. VALIDATION STEP: Predict the validation epoch using E1 and E2
    val_doublings = math.log2(e_val / e1)
    if val_doublings > 0:
        val_gains = (base_bit * 2) * (1 - (0.5 ** val_doublings))
    else:
        val_gains = 0
        
    val_pred = v1 + val_gains
    margin_of_error = abs(val_pred - v_val)
    
    print("\n--- VALIDATION RESULTS ---")
    print(f"Predicted Acc at Epoch {int(e_val)}: {val_pred:.3f}%")
    print(f"Actual Acc at Epoch {int(e_val)}:    {v_val:.3f}%")
    print(f"Calculated Margin of Error: +/- {margin_of_error:.3f}%")
    
    # 5. PREDICT FUTURE STEP: Predict the future epoch (E_future)
    future_doublings = math.log2(e_future / e2)
    if future_doublings > 0:
        future_gains = (base_bit * 2) * (1 - (0.5 ** future_doublings))
    else:
        future_gains = 0
        
    future_pred = v2 + future_gains
    
    # Apply upper and lower bounds using the margin of error
    lower_bound = future_pred - margin_of_error
    upper_bound = future_pred + margin_of_error
    
    print("\n--- FINAL PROJECTION RESULTS ---")
    print(f"Predicted Test Accuracy at Epoch {int(e_future)}: {future_pred:.3f}%")
    print(f"Confidence Range: {lower_bound:.3f}% to {upper_bound:.3f}%")
    print("(Range calculated using your 3-point validation margin!)")

if __name__ == "__main__":
    calculate_with_validation()