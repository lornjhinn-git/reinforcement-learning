def test_create_model_id():

    # package
    import uuid 
    from datetime import datetime

    # Generate a UUID
    unique_id = uuid.uuid4()

    # Get the current date and time
    current_datetime = datetime.now()

    # Format the date and time as a string in a timestamp-like manner (YYYYMMDDHHMMSS)
    timestamp_str = current_datetime.strftime("%Y%m%d%H%M%S")

    # Combine the UUID and timestamp to create a unique ID
    combined_id = f"{timestamp_str}_{unique_id}"

    print(combined_id)