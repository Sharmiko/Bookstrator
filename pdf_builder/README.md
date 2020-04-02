# How to use PDF builder

## Run the following command with correct argument values

```
python pdf_builder.py --csv_file=X --html_template=X --output_name=X
```
</br>

Arguments | Description
----------|------------
--csv_file | path/location of the input csv file</br>
--html_template | path/location of the input html template file</br>
--output_name | Output pdf file name</br>
</br>

### Note: input csv file should contain 3 columns
#### column for title, body text and corresponding image location


## Example


```
python convert_to_csv.py --csv_file="aesop.csv" --html_template="pdf_template" --output_name="output_file.pdf"
```
