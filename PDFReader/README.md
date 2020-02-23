# How to use Pdf to csv converter

## Run the following command with correct argument values

```
python convert_to_csv.py --input_file=X --output_file=X --title_size=X --body_size=X --start_page=X --end_page=X
```
</br>

--input_file - path/location of the input pdf file</br>
--output_file - path/location of the output csv file</br>
--title_size - font size of the title on the content page, default value 20</br>
--body_size - font size of the body on the content page, default value 10</br>
--start_page - starting page, where to began conversion from</br>
--end_page - page number where to end pdf conversion</br>

## Example


```
python convert_to_csv.py --input_file="aesop.pdf" --output_file="aesop_out.csv" --title_size=20 --body_size=10 --start_page=2 --end_page=88
```

In this case program opens `aesop.pdf` file, start conversion from the second page and ends conversion on 88th page, title and body font size are specified, and finally result is saved into `aesop_out.csv`
