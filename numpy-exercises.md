# NumPy Practice

This notebook offers a set of exercises for different tasks with NumPy.

It should be noted there may be more than one different way to answer a question or complete an exercise.

Exercises are based off (and directly taken from) the quick introduction to NumPy notebook.

Different tasks will be detailed by comments or text.

For further reference and resources, it's advised to check out the [NumPy documentation](https://numpy.org/devdocs/user/index.html).

And if you get stuck, try searching for a question in the following format: "how to do XYZ with numpy", where XYZ is the function you want to leverage from NumPy.


```python
# Import NumPy as its abbreviation 'np'
import numpy as np
```


```python
# Create a 1-dimensional NumPy array using np.array()
a = np.array([1,2,3])

# Create a 2-dimensional NumPy array using np.array()
b = np.array([[1,2,3],[4,5,6]])

# Create a 3-dimensional Numpy array using np.array()
c = np.array([[[1,2,3],[3,4,5],[4,5,6]],[[5,6,7],[8,9,10],[11,12,13]]])
(a,b,c)
```




    (array([1, 2, 3]),
     array([[1, 2, 3],
            [4, 5, 6]]),
     array([[[ 1,  2,  3],
             [ 3,  4,  5],
             [ 4,  5,  6]],
     
            [[ 5,  6,  7],
             [ 8,  9, 10],
             [11, 12, 13]]]))



Now we've you've created 3 different arrays, let's find details about them.

Find the shape, number of dimensions, data type, size and type of each array.

## type() function and dtype method

If you want to know what type of object it is, you will use the type() function to learn about it.

Now you know the object is a NumPy ndarray. If you want to know the type of data present in that array, you will use the attribute ‘.dtype’.


```python
# Attributes of 1-dimensional array (shape, 
# number of dimensions, data type, size and type)
(a.shape, a.ndim,a.dtype,a.size,type(a))
```




    ((3,), 1, dtype('int32'), 3, numpy.ndarray)




```python
# Attributes of 2-dimensional array
(b.shape,b.ndim,b.dtype,b.size,type(b))
```




    ((2, 3), 2, dtype('int32'), 6, numpy.ndarray)




```python
# Attributes of 3-dimensional array
(c.shape, c.ndim, c.dtype, c.size, type(c))
```




    ((2, 3, 3), 3, dtype('int32'), 18, numpy.ndarray)




```python
# Import pandas and create a DataFrame out of one
# of the arrays you've created
import pandas as pd
work_data = pd.DataFrame(b, columns=["Ram","Shyam","Ramu"], index=["Mon","Tues"])
work_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Ram</th>
      <th>Shyam</th>
      <th>Ramu</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Mon</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Tues</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create an array of shape (10, 2) with only ones
ones = np.ones(shape=(10,2))
ones
```




    array([[1., 1.],
           [1., 1.],
           [1., 1.],
           [1., 1.],
           [1., 1.],
           [1., 1.],
           [1., 1.],
           [1., 1.],
           [1., 1.],
           [1., 1.]])




```python
# Create an array of shape (7, 2, 3) of only zeros
zeros = np.zeros(shape=(7,2,3))
zeros
```




    array([[[0., 0., 0.],
            [0., 0., 0.]],
    
           [[0., 0., 0.],
            [0., 0., 0.]],
    
           [[0., 0., 0.],
            [0., 0., 0.]],
    
           [[0., 0., 0.],
            [0., 0., 0.]],
    
           [[0., 0., 0.],
            [0., 0., 0.]],
    
           [[0., 0., 0.],
            [0., 0., 0.]],
    
           [[0., 0., 0.],
            [0., 0., 0.]]])




```python
# Create an array within a range of 0 and 100 with step 3
f = np.arange(start=0,stop=100,step=3)
r = np.array([element for element in range(0,100,3)])
(f,r)
```




    (array([ 0,  3,  6,  9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48,
            51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99]),
     array([ 0,  3,  6,  9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48,
            51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99]))




```python
# Create a random array with numbers between 0 and 10 of size (7, 2)
random_array = np.random.randint(1,10,size=(7,2))
random_array
```




    array([[9, 9],
           [7, 9],
           [8, 9],
           [3, 3],
           [6, 5],
           [9, 9],
           [8, 4]])




```python
# Create a random array of floats between 0 & 1 of shape (3, 5)
random_float_array = np.random.rand(3,5)
random_float_array
```




    array([[0.2826893 , 0.60684747, 0.03517878, 0.70335721, 0.57414516],
           [0.39488601, 0.89657213, 0.613003  , 0.08878704, 0.97270814],
           [0.35017646, 0.04960241, 0.95605419, 0.50051158, 0.5709117 ]])




```python
# Set the random seed to 42
np.random.seed(42)

# Create a random array of numbers between 0 & 10 of size (4, 6)
random_natural_array = np.random.randint(1,10, size=(4,6))
random_natural_array
```




    array([[7, 4, 8, 5, 7, 3],
           [7, 8, 5, 4, 8, 8],
           [3, 6, 5, 2, 8, 6],
           [2, 5, 1, 6, 9, 1]])



Run the cell above again, what happens?

Are the numbers in the array different or the same? Why do think this is?

No the array is not different. It is because we fixed the seed. As we known that random does not generate the actual random number, it generate a pseudo number generated by the help of the algorithm which make use of seed value, If no value is passed,it automatically takes current time as a reference and generate number according to it.
And in the above example, we passed the value(42) to seed, due to which it is generating fixed array everytime. 


```python
# Create an array of random numbers between 1 & 10 of size (3, 7)
# and save it to a variable
random = np.random.randint(2,10, size=(3,7))
random
# Find the unique numbers in the array you just created
(random,np.unique(random))
```




    (array([[5, 7, 6, 8, 8, 6, 8],
            [4, 6, 5, 6, 9, 8, 4],
            [4, 7, 5, 3, 3, 6, 7]]),
     array([3, 4, 5, 6, 7, 8, 9]))




```python
# Find the 0'th index of the latest array you created
random[0]
```




    array([5, 7, 6, 8, 8, 6, 8])




```python
# Get the first 2 rows of latest array you created
random[:2]
```




    array([[5, 7, 6, 8, 8, 6, 8],
           [4, 6, 5, 6, 9, 8, 4]])




```python
# Get the first 2 values of the first 2 rows of the latest array
random[:2,:2] 

## To access the 2-D array, you need to use commas to separate the integers
## which represent the dimension and the index of the element. 
## The first integer represents the row and the other represents the column
```




    array([[5, 7],
           [4, 6]])




```python
# Create a random array of numbers between 0 & 10 and an array of ones
# both of size (3, 5), save them both to variables
random_array = np.random.randint(1,10, size=(3,5))
one_array = np.ones(shape=(3,5))
```


```python
# Add the two arrays together
sum = random_array + one_array
```


```python
# Create another array of ones of shape (5, 3)
another_array = np.ones(shape=(5,3))
```


```python
# Try add the array of ones and the other most recent array together
new_sum = sum + another_array.T
```

When you try the last cell, it produces an error. Why do think this is?

How would you fix it?
The error was due to brodcast (meaning both the array should have the same shape or one of the array should be one).
The term broadcasting refers to the ability of NumPy to treat arrays with different dimensions during arithmetic operations. This process involves certain rules that allow the smaller array to be ‘broadcast’ across the larger one, ensuring that they have compatible shapes for these operations.
1.Both the array should have the same shape
2.One array should have ones in there dimension meaning shape must (1,) or (,1)


```python
# Create another array of ones of shape (3, 5)
arr = np.ones(shape=(3,5))
arr
```




    array([[1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1.]])




```python
# Subtract the new array of ones from the other most recent array
diff = new_sum - arr
diff
```




    array([[10.,  6.,  7.,  5.,  8.],
           [10.,  8.,  2.,  2., 10.],
           [10.,  5., 10.,  4.,  8.]])




```python
# Multiply the ones array with the latest array
mult = diff * arr
mult
```




    array([[10.,  6.,  7.,  5.,  8.],
           [10.,  8.,  2.,  2., 10.],
           [10.,  5., 10.,  4.,  8.]])




```python
# Take the latest array to the power of 2 using '**'
pow = mult ** 2
pow
```




    array([[100.,  36.,  49.,  25.,  64.],
           [100.,  64.,   4.,   4., 100.],
           [100.,  25., 100.,  16.,  64.]])




```python
# Do the same thing with np.square()
np.square(mult)
```




    array([[100.,  36.,  49.,  25.,  64.],
           [100.,  64.,   4.,   4., 100.],
           [100.,  25., 100.,  16.,  64.]])




```python
# Find the mean of the latest array using np.mean()
np.mean(mult)
```




    7.0




```python
# Find the maximum of the latest array using np.max()
np.max(mult)
```




    10.0




```python
# Find the minimum of the latest array using np.min()
np.min(mult)
```




    2.0




```python
# Find the standard deviation of the latest array
np.std(mult)
```




    2.780887148615228




```python
# Find the variance of the latest array
np.var(mult)
```




    7.733333333333333




```python
# Reshape the latest array to (3, 5, 1)
reshape = np.reshape(mult,(3,5,1))
reshape
```




    array([[[10.],
            [ 6.],
            [ 7.],
            [ 5.],
            [ 8.]],
    
           [[10.],
            [ 8.],
            [ 2.],
            [ 2.],
            [10.]],
    
           [[10.],
            [ 5.],
            [10.],
            [ 4.],
            [ 8.]]])




```python
# Transpose the latest array
reshape.T
```




    array([[[10., 10., 10.],
            [ 6.,  8.,  5.],
            [ 7.,  2., 10.],
            [ 5.,  2.,  4.],
            [ 8., 10.,  8.]]])



What does the transpose do?
Inverse the rows and columns meaning columns value get interchange by rows values


```python
# Create two arrays of random integers between 0 to 10
# one of size (3, 3) the other of size (3, 2)
np.random.seed(0)
a = np.random.randint(10,size=(3,3))
b = np.random.randint(10, size=(3,2))
(a,b)
```




    (array([[5, 0, 3],
            [3, 7, 9],
            [3, 5, 2]]),
     array([[4, 7],
            [6, 8],
            [8, 1]]))




```python
# Perform a dot product on the two newest arrays you created
np.dot(a,b)
```




    array([[ 44,  38],
           [126,  86],
           [ 58,  63]])




```python
# Create two arrays of random integers between 0 to 10
# both of size (4, 3)
c= np.random.randint(10,size=(4,3))
d= np.random.randint(10,size=(4,3))
```


```python
# Perform a dot product on the two newest arrays you created
np.dot(c,d)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Cell In[131], line 2
          1 # Perform a dot product on the two newest arrays you created
    ----> 2 np.dot(c,d)
    

    ValueError: shapes (4,3) and (4,3) not aligned: 3 (dim 1) != 4 (dim 0)


It doesn't work. How would you fix it?


```python
# Take the latest two arrays, perform a transpose on one of them and then perform 
# a dot product on them both
# Shape not aligned error
np.dot(c,d.T)
```




    array([[ 94, 114, 118,  26],
           [ 55,  60,  79,  11],
           [ 57,  62,  74,  33],
           [  8,  12,   2,  16]])



Notice how performing a transpose allows the dot product to happen.

Why is this?

Checking out the documentation on [`np.dot()`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html) may help, as well as reading [Math is Fun's guide on the dot product](https://www.mathsisfun.com/algebra/vectors-dot-product.html).

Let's now compare arrays.


```python
# Create two arrays of random integers between 0 & 10 of the same shape
# and save them to variables
np.random.seed(0)
a = np.random.randint(1,10,size=(3,4))
b = np.random.randint(1,10,size=(3,4))
```


```python
# Compare the two arrays with '>'
a > b
```




    array([[False, False, False, False],
           [False, False,  True, False],
           [False,  True,  True,  True]])



What happens when you compare the arrays with `>`?
returns a new array containing the boolean value found by comparing elements at each rows and columns


```python
# Compare the two arrays with '>='
a>=b
```




    array([[False, False, False, False],
           [ True, False,  True, False],
           [False,  True,  True,  True]])




```python
a
```




    array([[6, 1, 4, 4],
           [8, 4, 6, 3],
           [5, 8, 7, 9]])




```python
# Find which elements of the first array are greater than 7
np.max(a)
```




    9




```python
# Which parts of each array are equal? (try using '==')
a == b
```




    array([[False, False, False, False],
           [ True, False, False, False],
           [False, False, False, False]])




```python
a
```




    array([[6, 1, 4, 4],
           [8, 4, 6, 3],
           [5, 8, 7, 9]])




```python
# Sort one of the arrays you just created in ascending order
np.sort(a)
```




    array([[1, 4, 4, 6],
           [3, 4, 6, 8],
           [5, 7, 8, 9]])




```python
# Sort the indexes of one of the arrays you just created
np.argsort(a)
```




    array([[1, 2, 3, 0],
           [3, 1, 2, 0],
           [0, 2, 1, 3]], dtype=int64)




```python
# Find the index with the maximum value in one of the arrays you've created
np.argmax(a)
```




    11




```python
# Find the index with the minimum value in one of the arrays you've created
np.argmin(a)
```




    1




```python
a
```




    array([[6, 1, 4, 4],
           [8, 4, 6, 3],
           [5, 8, 7, 9]])




```python
# Find the indexes with the maximum values down the 1st axis (axis=1)
# of one of the arrays you created
np.argmax(a, axis=1) ## axis =1 means along the row
```




    array([0, 0, 3], dtype=int64)




```python
# Find the indexes with the minimum values across the 0th axis (axis=0)
# of one of the arrays you created
np.argmin(a, axis=0) ## axis = 0 means along the columns
```




    array([2, 0, 0, 1], dtype=int64)




```python
# Create an array of normally distributed random numbers
rn = np.random.rand(2,3)
rn
```




    array([[0.87001215, 0.97861834, 0.79915856],
           [0.46147936, 0.78052918, 0.11827443]])




```python
# Create an array with 10 evenly spaced numbers between 1 and 100
num = np.linspace(1,100,num=10, dtype="int")
num
a = np.arange(1,100,11)
(num,a)
```




    (array([  1,  12,  23,  34,  45,  56,  67,  78,  89, 100]),
     array([ 1, 12, 23, 34, 45, 56, 67, 78, 89]))



## Extensions

For more exercises, check out the [NumPy quickstart tutorial](https://numpy.org/doc/stable/user/quickstart.html). A good practice would be to read through it and for the parts you find interesting, add them into the end of this notebook.

Pay particular attention to the section on broadcasting. And most importantly, get hands-on with the code as much as possible. If in dobut, run the code, see what it does.

The next place you could go is the [Stack Overflow page for the top questions and answers for NumPy](https://stackoverflow.com/questions/tagged/numpy?sort=MostVotes&edited=true). Often, you'll find some of the most common and useful NumPy functions here. Don't forget to play around with the filters! You'll likely find something helpful here.

Finally, as always, remember, the best way to learn something new is to try it. And try it relentlessly. If you get interested in some kind of NumPy function, asking yourself, "I wonder if NumPy could do that?", go and find out.
