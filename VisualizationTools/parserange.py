import pandas as pd

def parse_range_list(ordered_list,
                     full_dataframe):
        """
        A function that takes a full_dataframe, which has columns with titles.
        Ordered_list is a list of the form [1, 3-6, 9, 11, 17-21, etc.]
        The purpose of this function is to take the ordered list, and return
        a new list that includes every column title found at the index positions
        listed in the ordered list.
        """
        # Create a list to hold the column names
        column_names = []
        # Iterate through the ordered list
        for item in ordered_list:
            # Try to convert the item to an integer. This will be successful if it is a single entry
            try:
                # If successful, append the column name to the list
                column_names.append(full_dataframe.columns[item])
            # If it is not a single entry, it will be a range
            except TypeError:
                # Split the range into a list of two integers
                start, end = item.split('-')
                # Convert the integers to integers
                start = int(start)
                end = int(end)
                # Append the column names to the list
                column_names.extend(full_dataframe.columns[start:end+1])
        # Return the list of column names
        return column_names





            