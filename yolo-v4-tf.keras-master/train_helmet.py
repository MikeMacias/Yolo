# Define your class names as a list
class_names = ['person', 'car', 'tree', 'dog']

# Write class names to a text file
with open('class_names.txt', 'w') as f:
    for class_name in class_names:
        f.write(class_name + '\n')