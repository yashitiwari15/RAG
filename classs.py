# # Input number of rows
# m = int(input())

# # Input number of columns
# n = int(input())

# # Initialize count for non-zero elements
# non_zero_count = 0

# # Read the elements row by row
# for _ in range(m):
#     for _ in range(n):
#         value = int(input())
#         if value != 0:
#             non_zero_count += 1

# # Output the count of non-zero elements
# print(non_zero_count)

# Input number of rows
# Read the number of elements

# Read the number of elements
n = int(input())

# Initialize a list to store the elements
elements = []

# Read each element from the input
for _ in range(n):
    elements.append(int(input()))

# Initialize variable to store peak element
peak_element = 0

# Check for peak elements
if n == 1:
    # If there's only one element, it's the peak
    peak_element = elements[0]
else:
    for i in range(n):
        if i == 0:
            # First element, check against the second element
            if elements[i] > elements[i + 1]:
                peak_element = elements[i]
                break
        elif i == n - 1:
            # Last element, check against the second last element
            if elements[i] > elements[i - 1]:
                peak_element = elements[i]
                break
        else:
            # Middle elements, check against both previous and next elements
            if elements[i] > elements[i - 1] and elements[i] > elements[i + 1]:
                peak_element = elements[i]
                break

# Output the peak element or 0 if no peak is found
print(peak_element)
