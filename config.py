from os.path import dirname, abspath

d = dirname(dirname(abspath(__file__)))
d = d + '/'
# dir_of_file = dirname(d)
# parent_dir_of_file = dirname(dir_of_file)
#parents_parent_dir_of_file = dirname(parent_dir_of_file)

DATA_CONFIG = {
    'data_folder': '/home/reuben/newtest/',
    'project_folder': d}
