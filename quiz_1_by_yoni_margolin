def bubble_counter(arr):  # counting number of switches bubble sort
    n = len(arr)
    counter = 0
    for i in range(0, n - 1):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j+1]:
                switch(arr, j)  # switching
                counter += 1  # counting switches
    print(arr)
    print('counter = ' + str(counter))


def switch(arr, j):
    temp = arr[j]
    arr[j] = arr[j+1]
    arr[j+1] = temp


def bubble_counter_2(arr):  # counting number of switches by counting number of indexes right of every index which are
    # larger
    counter = 0
    n = len(arr)
    for i in range(0, n-1):
        for j in range(i+1, n):
            if arr[j] < arr[i]:
                counter = counter + 1
    print('counter 2 = ', counter)



def merge_counter(arr):  # counting number of switches in merge sort
    if len(arr) < 2:  # boundary condition
        return 0, arr
    middle = len(arr) // 2
    left = arr[:middle]
    right = arr[middle:]
    left_counter, left_ordered = merge_counter(left)  # sorting left array
    right_counter, right_ordered = merge_counter(right)  # sorting right array
    counter = left_counter + right_counter  # adding switches in the two sorts
    i = j = k = 0
    while i < len(left_ordered) and j < len(right_ordered):  # merging the two sorted arrays
        if left_ordered[i] < right_ordered[j]:
            arr[k] = left_ordered[i]
            i += 1
        else:
            arr[k] = right_ordered[j]

            counter += 1                        # count as single switch - correct
            # counter += len(left_ordered) - i  # count number of indexes which were jumped over - not correct

            j += 1
        k += 1
    while i < len(left_ordered):  # appending the ends of the sorted arrays
        arr[k] = left_ordered[i]
        i += 1
        k += 1
    while j < len(right_ordered):
        arr[k] = right_ordered[j]
        j += 1
        k += 1
    return counter, arr


# להכניס את המערך של מיון בועות פה בשורה מתחת של arr_for_bublle

arr_for_bublle=[ 67, 71, 65, 73, 34, 74, 24, 82, 90, 3, 100, 46, 72, 39, 80, 76, 23, 56, 86, 83]
bubble_counter(arr_for_bublle)
print("\n")

# להכניס את המערך של מיון בועות פה בשורה מתחת של arr_for_bublle_2

arr_for_bublle_2=[ 67, 71, 65, 73, 34, 74, 24, 82, 90, 3, 100, 46, 72, 39, 80, 76, 23, 56, 86, 83]
bubble_counter_2(arr_for_bublle_2)

# להכניס את המערך של מיון מיזוג פה בשורה מתחת של arr_for_merge
arr_for_merge=[968, 357, 255, 734, 396, 147, 815, 919, 121, 394, 333, 426, 364, 855, 219, 363, 930, 580, 658, 911, 986]
values_from_merge=merge_counter(arr_for_merge)
print(values_from_merge)
#הפלט הוא מספר - זה הקאונטר שלכם לתשובה במודל, ושאר המספרים הם המערך הממויין
