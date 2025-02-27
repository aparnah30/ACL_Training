def level_messages(message, level = 'INFO'):
    try:
        if len(message) > 30:
            raise
        return f"[{level}] {message} "
    except:
        print("Message to long")

print(level_messages("wrong input ", "ERROR"))
print(level_messages("Zero div errorrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr "))