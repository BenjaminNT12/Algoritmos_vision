"""
You are given an array of logs. Each log is a space-delimited string of words, where the first word is the identifier.

There are two types of logs:

Letter-logs: All words (except the identifier) consist of lowercase English letters.
Digit-logs: All words (except the identifier) consist of digits.

Reorder these logs so that:

The letter-logs come before all digit-logs.
The letter-logs are sorted lexicographically by their contents. If their contents are the same, then sort them lexicographically by their identifiers.
The digit-logs maintain their relative ordering.

Return the final order of the logs.
"""

class Solution:
    # @param A : list of strings
    # @return a list of strings
    def reorderLogs(self, A):
        digit_logs = []
        letter_logs = []
        for log in A:
            print(log)
            print(log.split("-")[1])
            if log.split("-")[1].isdigit():
                print("digit")
                digit_logs.append(log)
            else:
                letter_logs.append(log)
        # letter_logs.sort(key=lambda x: x.split()[0])
        # letter_logs.sort(key=lambda x: x.split()[1:])
        # return letter_logs + digit_logs
    
A = ["dig1-8-1-5-1", "let1-art-can", "dig2-3-6", "let2-own-kit-dig", "let3-art-zero"]

print(Solution().reorderLogs(A))