def sort(arr):
    if len(arr) < 10:
        return insertion_sort(arr)
    return quicksort(arr)
