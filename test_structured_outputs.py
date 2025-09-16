#!/usr/bin/env python3
"""
Test structured outputs functionality for GroqChat.
"""

import json
from dotenv import load_dotenv
from groq_chat import GroqChat

# Load environment variables
load_dotenv()

def test_json_schema_mode():
    """Test structured outputs using json_schema mode."""
    print("=== Testing JSON Schema Mode ===")
    
    # Define a JSON schema for person information
    person_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Full name of the person"},
            "age": {"type": "integer", "description": "Age in years"},
            "city": {"type": "string", "description": "City where they live"},
            "occupation": {"type": "string", "description": "Job or profession"},
            "hobbies": {
                "type": "array", 
                "items": {"type": "string"},
                "description": "List of hobbies"
            }
        },
        "required": ["name", "age", "city"],
        "additionalProperties": False
    }
    
    try:
        chat = GroqChat()
        
        # Test with structured output
        result = chat.invoke(
            "Generate information for a fictional person named Sarah who is 28 years old, lives in Seattle, works as a software engineer, and enjoys hiking and photography.",
            json_schema=person_schema
        )
        
        print(f"Query: Generate person information for Sarah...")
        print(f"Schema used: {json.dumps(person_schema, indent=2)}")
        
        if isinstance(result, dict):
            print("✓ Structured output received:")
            print(json.dumps(result, indent=2))
            
            # Validate required fields
            required_fields = person_schema["required"]
            missing_fields = [field for field in required_fields if field not in result]
            
            if not missing_fields:
                print("✓ All required fields present")
            else:
                print(f"✗ Missing required fields: {missing_fields}")
                
        else:
            print(f"✗ Expected dict, got: {type(result)} - {result}")
            
    except Exception as e:
        print(f"✗ JSON schema test failed: {e}")

def test_product_extraction():
    """Test extracting product information with structured output."""
    print("\n=== Testing Product Information Extraction ===")
    
    product_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Product name"},
            "price": {"type": "number", "description": "Price in USD"},
            "category": {"type": "string", "description": "Product category"},
            "features": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Key features"
            },
            "in_stock": {"type": "boolean", "description": "Availability status"}
        },
        "required": ["name", "price", "category"],
        "additionalProperties": False
    }
    
    try:
        chat = GroqChat()
        chat.clear_conversation_history()
        
        product_description = """
        The UltraBook Pro X1 is a premium laptop priced at $1299.99. 
        It's in the electronics category and features a 16-inch 4K display, 
        32GB RAM, 1TB SSD storage, and 12-hour battery life. 
        Currently available in stock.
        """
        
        result = chat.invoke(
            f"Extract product information from this description: {product_description}",
            json_schema=product_schema
        )
        
        print(f"Description: {product_description.strip()}")
        
        if isinstance(result, dict):
            print("✓ Product information extracted:")
            print(json.dumps(result, indent=2))
        else:
            print(f"✗ Expected dict, got: {result}")
            
    except Exception as e:
        print(f"✗ Product extraction test failed: {e}")

def test_data_analysis_schema():
    """Test structured output for data analysis."""
    print("\n=== Testing Data Analysis Schema ===")
    
    analysis_schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "Brief summary of findings"},
            "key_metrics": {
                "type": "object",
                "properties": {
                    "total": {"type": "integer"},
                    "average": {"type": "number"},
                    "highest": {"type": "number"},
                    "lowest": {"type": "number"}
                },
                "required": ["total", "average"]
            },
            "trends": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Observed trends"
            },
            "recommendations": {
                "type": "array", 
                "items": {"type": "string"},
                "description": "Action recommendations"
            }
        },
        "required": ["summary", "key_metrics"],
        "additionalProperties": False
    }
    
    try:
        chat = GroqChat()
        chat.clear_conversation_history()
        
        data_prompt = """
        Analyze this sales data: 
        Q1: $45,000, Q2: $52,000, Q3: $48,000, Q4: $61,000
        Total annual sales: $206,000
        """
        
        result = chat.invoke(
            f"Perform analysis on this sales data and provide insights: {data_prompt}",
            json_schema=analysis_schema
        )
        
        print(f"Data: {data_prompt.strip()}")
        
        if isinstance(result, dict):
            print("✓ Analysis completed:")
            print(json.dumps(result, indent=2))
        else:
            print(f"✗ Expected dict, got: {result}")
            
    except Exception as e:
        print(f"✗ Data analysis test failed: {e}")

def test_nested_schema():
    """Test complex nested schema structure."""
    print("\n=== Testing Complex Nested Schema ===")
    
    company_schema = {
        "type": "object",
        "properties": {
            "company_name": {"type": "string"},
            "founded_year": {"type": "integer"},
            "headquarters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "country": {"type": "string"},
                    "address": {"type": "string"}
                },
                "required": ["city", "country"]
            },
            "employees": {
                "type": "array",
                "items": {
                    "type": "object", 
                    "properties": {
                        "name": {"type": "string"},
                        "position": {"type": "string"},
                        "department": {"type": "string"}
                    },
                    "required": ["name", "position"]
                }
            },
            "revenue_millions": {"type": "number"}
        },
        "required": ["company_name", "founded_year", "headquarters"],
        "additionalProperties": False
    }
    
    try:
        chat = GroqChat()
        chat.clear_conversation_history()
        
        result = chat.invoke(
            "Create information for a fictional tech company called 'DataFlow Inc' founded in 2018, headquartered in Austin, Texas, with revenue of 45.2 million. Include 2-3 key employees.",
            json_schema=company_schema
        )
        
        print("Query: Create fictional tech company information...")
        
        if isinstance(result, dict):
            print("✓ Complex nested structure created:")
            print(json.dumps(result, indent=2))
        else:
            print(f"✗ Expected dict, got: {result}")
            
    except Exception as e:
        print(f"✗ Nested schema test failed: {e}")

def test_validation_and_error_handling():
    """Test validation and error handling with invalid schemas."""
    print("\n=== Testing Schema Validation ===")
    
    # Test with invalid schema (missing required properties)
    invalid_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"}
        }
        # Missing required field specification
    }
    
    try:
        chat = GroqChat()
        
        result = chat.invoke(
            "Generate a simple name object",
            json_schema=invalid_schema
        )
        
        print("✓ Invalid schema handled gracefully")
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"Schema validation error (expected): {e}")

def main():
    """Run all structured output tests."""
    print("GroqChat Structured Outputs Test Suite")
    print("=" * 50)
    
    test_json_schema_mode()
    test_product_extraction()
    test_data_analysis_schema()
    test_nested_schema()
    test_validation_and_error_handling()
    
    print("\n" + "=" * 50)
    print("Structured Outputs Test Suite Complete")

if __name__ == "__main__":
    main()