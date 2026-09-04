import math

def calculate_future_accuracy():
    print("=== AI DIMINISHING RETURNS CALCULATOR ===")
    
    # 1. Gather the known historical points
    print("\n--- Step 1: Input Baseline Data ---")
    e1 = float(input("Enter Baseline Epoch 1 (e.g., 150): "))
    v1 = float(input("Enter Test Accuracy at Baseline Epoch 1 (%): "))
    
    e2 = float(input("Enter Peak Epoch 2 (e.g., 250): "))
    v2 = float(input("Enter Test Accuracy at Peak Epoch 2 (%): "))
    
    # 2. Gather the prediction target
    print("\n--- Step 2: Input Target Goal ---")
    target_epoch = float(input("Enter the Target Epoch you want to predict (e.g., 25000): "))
    
    # 3. Calculate the actual progress bit seen between e1 and e2
    # Find how many doublings happened between the two check-ins
    doublings_between = math.log2(e2 / e1)
    
    # If doublings_between is 1 (like 150 to 300), the progress bit is exactly the difference.
    # Otherwise, we calculate the base progress bit (B) required to bridge that gap.
    actual_gap = v2 - v1
    
    # Geometric decay multiplier for the interval gap
    # This solves for the base bit 'B' where: Gap = B * (1 - 0.5^doublings) / (1 - 0.5)
    # Refined proxy for arbitrary step intervals matching your doubling curve:
    if doublings_between > 0:
        base_bit = actual_gap / (2 * (1 - (0.5 ** doublings_between)))
    else:
        base_bit = actual_gap
        
    print(f"\n[Calculated] Your Model's Real Progress Bit is: +{base_bit:.5f}%")
    print(f"[Calculated] Absolute Max Theoretical Ceiling: {(v2 + (base_bit * 2)):.3f}%")
    
    # 4. Project forward to the target epoch
    total_doublings_from_peak = math.log2(target_epoch / e2)
    
    if total_doublings_from_peak <= 0:
        # If looking backwards or exactly at peak
        predicted_v = v2
    else:
        # Calculate how much decay happens over the future doublings
        future_gains = (base_bit * 2) * (1 - (0.5 ** total_doublings_from_peak))
        predicted_v = v2 + future_gains

    print("\n--- PROJECTION RESULTS ---")
    print(f"Predicted Test Accuracy at Epoch {int(target_epoch)}: {predicted_v:.3f}%")

if __name__ == "__main__":
    calculate_future_accuracy()
