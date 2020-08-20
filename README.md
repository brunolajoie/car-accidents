# car-accidents

To run the script, you must have the following files stored in the same folder:
- caracteristics.csv
- users.csv
- places.csv
- vehicles.csv

Once it is done, open `script.py` and edit the global variable DATAPATH:

```
# for example
DATAPATH = /home/USER/DATAFOLDER/
```

Open a terminal and run the script (it can take a little while depending on your machine):

```
python script.py
```

## Benefits of packaging your code using pipelines

- Better understanding of the data flow
- Increased modularity
- Increased readability
- Increased maintainability
- Improved RAM management (less temporary variables)

## Drawbacks

- Need to start from scratch if the workflow drastically changes
- Need to be very careful with how steps are organized 
