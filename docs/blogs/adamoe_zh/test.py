from collections import Counter
def all_sort(nums):
    res = []
    track = []
    not_used = Counter(nums)
    backtrack(nums, res, track, not_used)
    return res

def backtrack(nums, res, track, not_used):
    if len(track) == len(nums):
        res.append(track[:])
        return
    for n in nums:
        if not_used[n] == 0:
            continue
        track.append(n)
        not_used[n] -= 1
        backtrack(nums, res, track, not_used)
        not_used[n] += 1
        track.pop()

nums = [1,2,3]

print(all_sort(nums))