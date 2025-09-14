"""
Utility functions for the AI Document Analyser project.
"""

import inspect
from typing import Callable, Optional


def get_function_docstring(func: Callable) -> str:
    """
    Extract and return the first paragraph of the docstring from a Python function.
    
    Args:
        func (Callable): The Python function to extract docstring from
        
    Returns:
        str: The first paragraph of the function's docstring, or empty string if no docstring exists
        
    Examples:
        >>> def example_func():
        ...     '''This is an example function.
        ...
        ...     More details here.'''
        ...     pass
        >>> get_function_docstring(example_func)
        'This is an example function.'
        
        >>> def no_docstring():
        ...     pass
        >>> get_function_docstring(no_docstring)
        ''
    """
    if not callable(func):
        raise TypeError(f"Expected a callable function, got {type(func).__name__}")
    
    docstring = inspect.getdoc(func) or ""
    # Split on double newlines and return the first part
    first_paragraph = docstring.split('\n\n', 1)[0].strip()
    return first_paragraph


def get_function_docstring_raw(func: Callable) -> Optional[str]:
    """
    Extract the raw docstring from a Python function without any processing.
    
    This function returns the exact docstring as it appears in the function,
    including any indentation and formatting, or None if no docstring exists.
    
    Args:
        func (Callable): The Python function to extract raw docstring from
        
    Returns:
        Optional[str]: The function's raw docstring, or None if no docstring exists
        
    Examples:
        >>> def example_func():
        ...     '''    This has indentation.
        ...     And multiple lines.
        ...     '''
        ...     pass
        >>> get_function_docstring_raw(example_func)
        '    This has indentation.\\n    And multiple lines.\\n    '
    """
    if not callable(func):
        raise TypeError(f"Expected a callable function, got {type(func).__name__}")
    
    # Access the raw __doc__ attribute
    return func.__doc__


def get_function_info(func: Callable) -> dict:
    """
    Get comprehensive information about a function including its docstring.
    
    Args:
        func (Callable): The Python function to analyze
        
    Returns:
        dict: Dictionary containing function information with keys:
            - name: Function name
            - docstring: Cleaned docstring
            - raw_docstring: Raw docstring with original formatting
            - signature: Function signature as string
            - parameters: List of parameter names
            - file: File where function is defined (if available)
            - line_number: Line number where function starts (if available)
            
    Examples:
        >>> def sample_func(x: int, y: str = "default") -> str:
        ...     '''A sample function for demonstration.'''
        ...     return f"{x}: {y}"
        >>> info = get_function_info(sample_func)
        >>> info['name']
        'sample_func'
        >>> info['docstring']
        'A sample function for demonstration.'
    """
    if not callable(func):
        raise TypeError(f"Expected a callable function, got {type(func).__name__}")
    
    # Get function signature
    try:
        signature = inspect.signature(func)
        signature_str = str(signature)
        parameters = list(signature.parameters.keys())
    except (ValueError, TypeError):
        signature_str = "Unable to determine signature"
        parameters = []
    
    # Get source file and line number
    try:
        source_file = inspect.getfile(func)
        line_number = inspect.getsourcelines(func)[1]
    except (OSError, TypeError):
        source_file = "Unknown"
        line_number = None
    
    return {
        'name': func.__name__,
        'docstring': get_function_docstring(func),
        'raw_docstring': get_function_docstring_raw(func),
        'signature': signature_str,
        'parameters': parameters,
        'file': source_file,
        'line_number': line_number
    }


def parse_string_or_dict(input_data):
    """
    Parse input that can be either a string containing a dictionary or a dictionary itself.
    
    If the input is already a dictionary, it returns the exact same dictionary.
    If the input is a string, it attempts to parse a dictionary from the string using
    multiple parsing strategies including JSON, Python literal evaluation, and regex patterns.
    
    Args:
        input_data (str | dict): Either a string containing a dictionary or a dictionary
        
    Returns:
        dict: The parsed dictionary or the original dictionary
        
    Raises:
        ValueError: If the input is a string but no valid dictionary can be parsed from it
        TypeError: If the input is neither a string nor a dictionary
        
    Examples:
        >>> parse_string_or_dict({'key': 'value'})
        {'key': 'value'}
        
        >>> parse_string_or_dict('{"name": "John", "age": 30}')
        {'name': 'John', 'age': 30}
        
        >>> parse_string_or_dict("{'language': 'Python', 'version': 3.9}")
        {'language': 'Python', 'version': 3.9}
        
        >>> parse_string_or_dict("Here's the data: {'result': 'success', 'count': 42}")
        {'result': 'success', 'count': 42}
    """
    import json
    import ast
    import re
    
    # If input is already a dictionary, return it as-is
    if isinstance(input_data, dict):
        return input_data
    
    # If input is not a string, raise error
    if not isinstance(input_data, str):
        raise TypeError(f"Expected string or dict, got {type(input_data).__name__}")
    
    # Clean the input string
    text = input_data.strip()
    
    # Strategy 1: Try direct JSON parsing
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Strategy 2: Try Python literal evaluation (for Python dict syntax)
    try:
        result = ast.literal_eval(text)
        if isinstance(result, dict):
            return result
    except (ValueError, SyntaxError):
        pass
    
    # Strategy 3: Extract dictionary using regex patterns
    dict_patterns = [
        r'\{[^{}]*\}',  # Basic pattern for simple dicts
        r'\{(?:[^{}]|{[^{}]*})*\}',  # Pattern for nested dicts (one level)
        r'\{(?:[^{}]|{(?:[^{}]|{[^{}]*})*})*\}',  # Pattern for deeper nesting
    ]
    
    for pattern in dict_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            # Try JSON parsing on the match
            try:
                result = json.loads(match)
                if isinstance(result, dict):
                    return result
            except (json.JSONDecodeError, ValueError):
                pass
            
            # Try Python literal evaluation on the match
            try:
                result = ast.literal_eval(match)
                if isinstance(result, dict):
                    return result
            except (ValueError, SyntaxError):
                pass
    
    # Strategy 4: Try to fix common formatting issues and retry
    # Replace single quotes with double quotes for JSON compatibility
    json_like = re.sub(r"'([^']*)':", r'"\1":', text)  # Keys
    json_like = re.sub(r":\s*'([^']*)'", r': "\1"', json_like)  # Values
    
    try:
        result = json.loads(json_like)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Strategy 5: Look for key-value patterns and construct dictionary
    # Pattern for key: value pairs
    kv_pattern = r'["\']?(\w+)["\']?\s*:\s*["\']?([^,}\]]+?)["\']?(?=\s*[,}]|$)'
    matches = re.findall(kv_pattern, text)
    
    if matches:
        result = {}
        for key, value in matches:
            # Try to convert value to appropriate type
            try:
                # Try numeric conversion
                if '.' in value:
                    result[key] = float(value)
                else:
                    result[key] = int(value)
            except ValueError:
                # Try boolean conversion
                if value.lower() in ('true', 'false'):
                    result[key] = value.lower() == 'true'
                else:
                    # Keep as string, removing quotes if present
                    result[key] = value.strip('"\'')
        
        if result:
            return result
    
    # If all strategies fail, raise an error
    raise ValueError(f"Could not parse dictionary from string: {text[:100]}...")


    def extract_tagged_content(text: str, tag: str) -> tuple[str, str]:
        """
        Extract content between <tag> and </tag> from a string, and also return the part after the closing tag.

        Args:
            text (str): The input string.
            tag (str): The tag name (without angle brackets).

        Returns:
            tuple[str, str]: (content between the tags, part of string after closing tag). If tags not found, returns ("", "").
        """
        opening = f"<{tag}>"
        closing = f"</{tag}>"
        start = text.find(opening)
        if start == -1:
            return "", ""
        start += len(opening)
        end = text.find(closing, start)
        if end == -1:
            return "", ""
        enclosed = text[start:end]
        after = text[end + len(closing):]
        return enclosed, after


# Example usage and testing
if __name__ == "__main__":
    def test_function(x: int, y: str = "hello") -> str:
        """
        This is a test function for demonstrating docstring extraction.
        
        Args:
            x (int): An integer parameter
            y (str): A string parameter with default value
            
        Returns:
            str: A formatted string combining x and y
            
        Example:
            >>> test_function(42, "world")
            '42: world'
        """
        return f"{x}: {y}"
    
    def no_docstring_func():
        pass
    
    # Test the functions
    print("=== Testing get_function_docstring ===")
    print(f"With docstring: '{get_function_docstring(test_function)}'")
    print(f"Without docstring: '{get_function_docstring(no_docstring_func)}'")
    
    print("\n=== Testing get_function_docstring_raw ===")
    print(f"Raw docstring: {repr(get_function_docstring_raw(test_function))}")
    print(f"Raw (no docstring): {get_function_docstring_raw(no_docstring_func)}")
    
    print("\n=== Testing get_function_info ===")
    info = get_function_info(test_function)
    for key, value in info.items():
        print(f"{key}: {value}")
    
    print("\n=== Testing parse_string_or_dict ===")
    
    # Test with dictionary input
    test_dict = {'name': 'Alice', 'age': 25, 'active': True}
    result1 = parse_string_or_dict(test_dict)
    print(f"Dict input: {test_dict}")
    print(f"Result: {result1}")
    print(f"Same object: {result1 is test_dict}")
    
    # Test with JSON string
    json_string = '{"language": "Python", "version": 3.9, "stable": true}'
    result2 = parse_string_or_dict(json_string)
    print(f"\nJSON string: {json_string}")
    print(f"Result: {result2}")
    
    # Test with Python dict string
    python_dict_string = "{'framework': 'FastAPI', 'async': True, 'port': 8000}"
    result3 = parse_string_or_dict(python_dict_string)
    print(f"\nPython dict string: {python_dict_string}")
    print(f"Result: {result3}")
    
    # Test with embedded dictionary in text
    text_with_dict = "The API returned this data: {'status': 'success', 'count': 42, 'data': 'processed'}"
    result4 = parse_string_or_dict(text_with_dict)
    print(f"\nText with embedded dict: {text_with_dict}")
    print(f"Result: {result4}")
    
    # Test with malformed but parseable string
    messy_string = "key1: value1, key2: 123, key3: true"
    try:
        result5 = parse_string_or_dict(messy_string)
        print(f"\nMessy string: {messy_string}")
        print(f"Result: {result5}")
    except ValueError as e:
        print(f"\nMessy string failed: {e}")
    
    # Test error cases
    try:
        parse_string_or_dict(123)
    except TypeError as e:
        print(f"\nError with integer input: {e}")
    
    try:
        parse_string_or_dict("This is just plain text with no dictionary")
    except ValueError as e:
        print(f"\nError with plain text: {e}")


def extract_tagged_content(text: str, tag: str) -> str:
    """
    Extract content between XML-like tags from a string.
    
    Args:
        text (str): The input string containing tagged content
        tag (str): The tag name (without angle brackets)
        
    Returns:
        str: The content between the opening and closing tags, or empty string if tags not found
        
    Examples:
        >>> extract_tagged_content("Hello <final_answer>World</final_answer>!", "final_answer")
        'World'
        
        >>> extract_tagged_content("Data: <result>42</result> End", "result")
        '42'
        
        >>> extract_tagged_content("No tags here", "missing")
        ''
        
        >>> extract_tagged_content("<answer>Multi\nline\ncontent</answer>", "answer")
        'Multi\nline\ncontent'
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected string input, got {type(text).__name__}")
    
    if not isinstance(tag, str):
        raise TypeError(f"Expected string tag, got {type(tag).__name__}")
    
    # Create opening and closing tag patterns
    opening_tag = f"<{tag}>"
    closing_tag = f"</{tag}>"
    
    # Find the start and end positions
    start_pos = text.find(opening_tag)
    if start_pos == -1:
        return ""  # Opening tag not found
    
    # Move past the opening tag
    content_start = start_pos + len(opening_tag)
    
    # Find the closing tag starting from after the opening tag
    end_pos = text.find(closing_tag, content_start)
    if end_pos == -1:
        return ""  # Closing tag not found
    
    # Extract and return the content between tags
    return text[content_start:end_pos]


if __name__ == "__main__":
    # Test the tag extraction function
    print("=== Testing extract_tagged_content function ===")
    
    # Test basic functionality
    test1 = "Hello <final_answer>World</final_answer>!"
    result1 = extract_tagged_content(test1, "final_answer")
    print(f"Input: {test1}")
    print(f"Tag: final_answer")
    print(f"Result: '{result1}'")
    
    # Test with different tag
    test2 = "Data: <result>42</result> End"
    result2 = extract_tagged_content(test2, "result")
    print(f"\nInput: {test2}")
    print(f"Tag: result")
    print(f"Result: '{result2}'")
    
    # Test with multiline content
    test3 = "<answer>Multi\nline\ncontent</answer>"
    result3 = extract_tagged_content(test3, "answer")
    print(f"\nInput: {test3}")
    print(f"Tag: answer")
    print(f"Result: '{result3}'")
    
    # Test with missing tags
    test4 = "No tags here"
    result4 = extract_tagged_content(test4, "missing")
    print(f"\nInput: {test4}")
    print(f"Tag: missing")
    print(f"Result: '{result4}'")
    
    # Test with only opening tag
    test5 = "Start <incomplete>content without closing"
    result5 = extract_tagged_content(test5, "incomplete")
    print(f"\nInput: {test5}")
    print(f"Tag: incomplete")
    print(f"Result: '{result5}'")
    
    print("\n=== Previous tests ===")
    
    # Previous test code for other functions
