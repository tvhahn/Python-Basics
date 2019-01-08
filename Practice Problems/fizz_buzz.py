def fizz_buzz(num):
    if num is None:
        raise TypeError('Not a valid number')
    if num < 1:
        raise ValueError('Cannot be less than 1')
    
    result = []
    for i in range(1, num+1):
        if i % 3 == 0 and i % 5 == 0:
            result.append('FizzBuzz')
        elif i % 3 == 0:
            result.append('Fizz')
        elif i % 5 == 0:
            result.append('Buzz')
        else:
            result.append(str(i))
    
    return(result)

print(fizz_buzz(15))