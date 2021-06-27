#Given string S and dictionary dict, check if S is made of substrings of the dictionary words.
#Algorithm explained in picures directory

def findWords(dict, s):
    list_length = []
    for word in dict:
        word_length = len(word)
        if not word_length in list_length and word_length <= len(s):
            list_length.append(word_length)
    # O(n)
    list_length.sort()
    # O(n*logn)
    return splitWords(dict, s, list_length)

def splitWords(dict, s, list_length):
    if not s:
        return True
    for lens in list_length:     # worst case - list_length has 's' cells
        for i in range(0, len(s) - lens + 1):    # worst case - len is 1 and i runs from 0 to 's'
            if s[i:i+lens] in dict:
                return (splitWords(dict, s[0:i], list_length) and splitWords(dict, s[i+lens:], list_length)) 
                # worst case - string size 's' splits to single letters so recursive runtime is 'n^2'
    return False           

#driver code
dict = []
for letter in range(ord('a'),ord('z')+1):
    dict.append(chr(letter))
print(findWords(dict,"oposum"))


                
    




