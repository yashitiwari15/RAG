def min_steps_to_divisible(arr, x):
    n = len(arr)
    moves = 0
    y = 1  # Start Y from 1
    incremented_values = set()

    while True:
        all_divisible = True
        for i in range(n):
            if arr[i] % x != 0:
                arr[i] += y
                incremented_values.add(y)
                y += 1
                moves += 1
                all_divisible = False
        if all_divisible:
            break
    return moves

# Function to take input and call the solution function
def main():
    n = int(input().strip())
    arr = list(map(int, input().strip().split()))
    x = int(input().strip())
    result = min_steps_to_divisible(arr, x)
    print(result)

# Test the function with provided test cases
main()
