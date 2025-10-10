from typing import List

class Solution:
    def kidswithCandies(self, candies: List[int], extraCandies: int) -> List[bool]:
        output = []
        max_candies = max(candies)
        for i in candies:
            if i + extraCandies >= max_candies:
                output.append(True)
            else:
                output.append(False)
        return output

solution = Solution()
result = solution.kidswithCandies([2, 3, 5, 1, 2], 3)
print(result)
