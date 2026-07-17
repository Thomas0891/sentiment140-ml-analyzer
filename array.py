from array import *

new = array('i', [])
n = int(input("Enter the number of elements in the array: "))
for i in range(0, n):
    new.append(int(input("Enter the elements of the array next: ")))
for x in new:
    print(x, end=" ")
