# Function to find the minimum time for each test case
def min_delivery_time(n, friends_time, mayur_time):
    min_times = []
    for i in range(n):
        # For each item, calculate the minimum time it takes (either by friend or Mayur)
        min_times.append(min(friends_time[i], mayur_time[i]))
    
    # The overall time will be the maximum of these minimum times
    return max(min_times)

# Main code to handle multiple test cases
def main():
    t = int(input())  # Number of test cases
    results = []  # List to store results for each test case
    
    for _ in range(t):
        n = int(input())  # Number of food items
        friends_time = list(map(int, input().split()))  # Time taken by friends to deliver each item
        mayur_time = list(map(int, input().split()))  # Time taken by Mayur to deliver each item
        
        # Calculate the minimum delivery time for this test case and store it in results
        results.append(min_delivery_time(n, friends_time, mayur_time))
    
    # After processing all test cases, print the results
    for result in results:
        print(result)

# Run the main function
if __name__ == "__main__":
    main()
