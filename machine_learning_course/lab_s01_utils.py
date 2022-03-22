import inspect


def print_function_name(number_of_empty_lines_before: int = 3) -> None:
    print('\n'*number_of_empty_lines_before)
    print(f'{inspect.stack()[1][3]}')
