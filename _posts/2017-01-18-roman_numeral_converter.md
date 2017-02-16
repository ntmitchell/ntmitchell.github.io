---
title: "Simple Roman Numeral Converter"
header:
  overlay_color: "#333"
date: 2017-01-03
tags: Python
---

Here's the code from a python programming exercise for class.

We were to write two functions that, when given any integer from 1 through 1000, could:

* Translate Roman numerals into the corresponding integers.
* Convert integers into Roman numerals.

Lastly, we wanted to verify that the functions work. We compared our outputs to a sample of "solved" numbers, and also fed the results from one function into the other to make sure they gave consistent results.


# Functions

The numerals-to-number converter looks at a character, determines if it's corresponding value is less than or greater than its neighbor, and determines if the value should be added or subtracted from the total. For example, XXIV becomes [10, 10, -1, 5], which sums to 24.

My number-to-numeral converter is a little more complicated. It relies on the fact that, aside from 1, individual Roman numerals are written to represent base-10 multiples of 5 or 10 (i.e., 5, 50, 500).

```python
def str_to_int(numeral_string):
    roman_numerals_dict = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D':500, 'M':1000}

    # Divide the numeral string into a list of characters
    numbers = []
    for numeral in numeral_string:
        numbers.append(roman_numerals_dict[numeral])

    for index in range(len(numbers) - 1):
        if numbers[index] < numbers[index + 1]:
            numbers[index] = numbers[index] * -1

    return(sum(numbers))
```


```python
def int_to_string(integer_number, index = 0):
    number_to_roman_nums_dict = {1: 'I', 5: 'V', 10: 'X', 50: 'L', 100:'C', 500: 'D', 1000: 'M'}

    result_string = str()

    # Convert the input values into a string
    integer_number_as_string = str(integer_number)
    number_of_digits = len(integer_number_as_string)

    # Determine the last numeral in the string
    ending_number = int(integer_number_as_string[-1])

    # Convert the last numeral into a roman numeral
    if ending_number < 4:
        # Ex. 3 -> resulting_string = 3 * "I" = "III"
        result_string = ending_number * number_to_roman_nums_dict[1 * (10 ** index)]

    if ending_number == 4:
        # Ex. 4 -> resulting_string = "I" + "V" = "IV" = 4
        result_string = number_to_roman_nums_dict[1 * (10 ** index)] + number_to_roman_nums_dict[5 * (10 ** index)]

    if ending_number > 4 and ending_number < 9:
        # Ex. 8 -> resulting_string = "V" + "I" + "I"= "VII" = 8
        result_string = number_to_roman_nums_dict[5 * (10 ** index)] + (ending_number % 5) * number_to_roman_nums_dict[1 * (10 ** index)]

    if ending_number == 9:
        # Ex. 90 -> ending_number = 9, index = 1.
        #     resulting_string = number_to_roman_nums_dict[1 * 10^1] + number_to_roman_nums_dict[10 * 10^1]
        #                      = number_to_roman_nums_dict[10] + number_to_roman_nums_dict[100]
        #                      = "X" + "C" = "XC"
        result_string = number_to_roman_nums_dict[1 * (10 ** index)] + number_to_roman_nums_dict[10 * (10 ** index)]

    # Use recursion to evaluate numerals in the tens, hundreds, and thousands places
    if number_of_digits > 1:
        index += 1
        result_string = int_to_string(int(integer_number_as_string[0:-1]), index) + result_string

    return(result_string)
```

# Unit testing


```python
from random import randint
import pandas as pd
from numpy import vectorize
```


```python
success = True
for number in range(1,1000):
    if str_to_int(int_to_string(number)) != number:
        success = False
        print("Failed on {}.".format(number))
if success:
    print("All numbers converted correctly.")
```

    All numbers converted correctly.



```python
# Randomly select 100 numbers from 1 to 1000
random_integers = [randint(1, 1000) for _ in range(100)]

dataframe = pd.DataFrame(random_integers, columns = ["integer value"])
dataframe["integers converted to roman numeral"] = dataframe["integer value"].apply(vectorize(int_to_string))
dataframe["numerals converted back to integer"] = dataframe["integers converted to roman numeral"].apply(vectorize(str_to_int))
print(dataframe)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>integer value</th>
      <th>integers converted to roman numeral</th>
      <th>numerals converted back to integer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>539</td>
      <td>DXXXIX</td>
      <td>539</td>
    </tr>
    <tr>
      <th>1</th>
      <td>353</td>
      <td>CCCLIII</td>
      <td>353</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1000</td>
      <td>M</td>
      <td>1000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>182</td>
      <td>CLXXXII</td>
      <td>182</td>
    </tr>
    <tr>
      <th>4</th>
      <td>239</td>
      <td>CCXXXIX</td>
      <td>239</td>
    </tr>
    <tr>
      <th>5</th>
      <td>869</td>
      <td>DCCCLXIX</td>
      <td>869</td>
    </tr>
    <tr>
      <th>6</th>
      <td>378</td>
      <td>CCCLXXVIII</td>
      <td>378</td>
    </tr>
    <tr>
      <th>7</th>
      <td>497</td>
      <td>CDXCVII</td>
      <td>497</td>
    </tr>
    <tr>
      <th>8</th>
      <td>980</td>
      <td>CMLXXX</td>
      <td>980</td>
    </tr>
    <tr>
      <th>9</th>
      <td>459</td>
      <td>CDLIX</td>
      <td>459</td>
    </tr>
    <tr>
      <th>10</th>
      <td>23</td>
      <td>XXIII</td>
      <td>23</td>
    </tr>
    <tr>
      <th>11</th>
      <td>146</td>
      <td>CXLVI</td>
      <td>146</td>
    </tr>
    <tr>
      <th>12</th>
      <td>427</td>
      <td>CDXXVII</td>
      <td>427</td>
    </tr>
    <tr>
      <th>13</th>
      <td>158</td>
      <td>CLVIII</td>
      <td>158</td>
    </tr>
    <tr>
      <th>14</th>
      <td>423</td>
      <td>CDXXIII</td>
      <td>423</td>
    </tr>
    <tr>
      <th>15</th>
      <td>669</td>
      <td>DCLXIX</td>
      <td>669</td>
    </tr>
    <tr>
      <th>16</th>
      <td>538</td>
      <td>DXXXVIII</td>
      <td>538</td>
    </tr>
    <tr>
      <th>17</th>
      <td>981</td>
      <td>CMLXXXI</td>
      <td>981</td>
    </tr>
    <tr>
      <th>18</th>
      <td>830</td>
      <td>DCCCXXX</td>
      <td>830</td>
    </tr>
    <tr>
      <th>19</th>
      <td>9</td>
      <td>IX</td>
      <td>9</td>
    </tr>
    <tr>
      <th>20</th>
      <td>21</td>
      <td>XXI</td>
      <td>21</td>
    </tr>
    <tr>
      <th>21</th>
      <td>170</td>
      <td>CLXX</td>
      <td>170</td>
    </tr>
    <tr>
      <th>22</th>
      <td>788</td>
      <td>DCCLXXXVIII</td>
      <td>788</td>
    </tr>
    <tr>
      <th>23</th>
      <td>718</td>
      <td>DCCXVIII</td>
      <td>718</td>
    </tr>
    <tr>
      <th>24</th>
      <td>200</td>
      <td>CC</td>
      <td>200</td>
    </tr>
    <tr>
      <th>25</th>
      <td>531</td>
      <td>DXXXI</td>
      <td>531</td>
    </tr>
    <tr>
      <th>26</th>
      <td>468</td>
      <td>CDLXVIII</td>
      <td>468</td>
    </tr>
    <tr>
      <th>27</th>
      <td>203</td>
      <td>CCIII</td>
      <td>203</td>
    </tr>
    <tr>
      <th>28</th>
      <td>642</td>
      <td>DCXLII</td>
      <td>642</td>
    </tr>
    <tr>
      <th>29</th>
      <td>312</td>
      <td>CCCXII</td>
      <td>312</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>70</th>
      <td>929</td>
      <td>CMXXIX</td>
      <td>929</td>
    </tr>
    <tr>
      <th>71</th>
      <td>648</td>
      <td>DCXLVIII</td>
      <td>648</td>
    </tr>
    <tr>
      <th>72</th>
      <td>26</td>
      <td>XXVI</td>
      <td>26</td>
    </tr>
    <tr>
      <th>73</th>
      <td>7</td>
      <td>VII</td>
      <td>7</td>
    </tr>
    <tr>
      <th>74</th>
      <td>246</td>
      <td>CCXLVI</td>
      <td>246</td>
    </tr>
    <tr>
      <th>75</th>
      <td>272</td>
      <td>CCLXXII</td>
      <td>272</td>
    </tr>
    <tr>
      <th>76</th>
      <td>261</td>
      <td>CCLXI</td>
      <td>261</td>
    </tr>
    <tr>
      <th>77</th>
      <td>4</td>
      <td>IV</td>
      <td>4</td>
    </tr>
    <tr>
      <th>78</th>
      <td>523</td>
      <td>DXXIII</td>
      <td>523</td>
    </tr>
    <tr>
      <th>79</th>
      <td>220</td>
      <td>CCXX</td>
      <td>220</td>
    </tr>
    <tr>
      <th>80</th>
      <td>134</td>
      <td>CXXXIV</td>
      <td>134</td>
    </tr>
    <tr>
      <th>81</th>
      <td>368</td>
      <td>CCCLXVIII</td>
      <td>368</td>
    </tr>
    <tr>
      <th>82</th>
      <td>983</td>
      <td>CMLXXXIII</td>
      <td>983</td>
    </tr>
    <tr>
      <th>83</th>
      <td>767</td>
      <td>DCCLXVII</td>
      <td>767</td>
    </tr>
    <tr>
      <th>84</th>
      <td>2</td>
      <td>II</td>
      <td>2</td>
    </tr>
    <tr>
      <th>85</th>
      <td>356</td>
      <td>CCCLVI</td>
      <td>356</td>
    </tr>
    <tr>
      <th>86</th>
      <td>355</td>
      <td>CCCLV</td>
      <td>355</td>
    </tr>
    <tr>
      <th>87</th>
      <td>499</td>
      <td>CDXCIX</td>
      <td>499</td>
    </tr>
    <tr>
      <th>88</th>
      <td>464</td>
      <td>CDLXIV</td>
      <td>464</td>
    </tr>
    <tr>
      <th>89</th>
      <td>786</td>
      <td>DCCLXXXVI</td>
      <td>786</td>
    </tr>
    <tr>
      <th>90</th>
      <td>975</td>
      <td>CMLXXV</td>
      <td>975</td>
    </tr>
    <tr>
      <th>91</th>
      <td>518</td>
      <td>DXVIII</td>
      <td>518</td>
    </tr>
    <tr>
      <th>92</th>
      <td>798</td>
      <td>DCCXCVIII</td>
      <td>798</td>
    </tr>
    <tr>
      <th>93</th>
      <td>414</td>
      <td>CDXIV</td>
      <td>414</td>
    </tr>
    <tr>
      <th>94</th>
      <td>835</td>
      <td>DCCCXXXV</td>
      <td>835</td>
    </tr>
    <tr>
      <th>95</th>
      <td>246</td>
      <td>CCXLVI</td>
      <td>246</td>
    </tr>
    <tr>
      <th>96</th>
      <td>243</td>
      <td>CCXLIII</td>
      <td>243</td>
    </tr>
    <tr>
      <th>97</th>
      <td>888</td>
      <td>DCCCLXXXVIII</td>
      <td>888</td>
    </tr>
    <tr>
      <th>98</th>
      <td>614</td>
      <td>DCXIV</td>
      <td>614</td>
    </tr>
    <tr>
      <th>99</th>
      <td>99</td>
      <td>XCIX</td>
      <td>99</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 3 columns</p>
</div>




```python

```
