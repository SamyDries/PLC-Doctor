
from gemini_assistant import generate_diagnosis

result = generate_diagnosis(
    fault_type="Motor Overload",
    probability=95.0,
    machine_data={
        "motor_current": 25.5,
        "motor_temperature": 95.0,
        "vibration": 4.2,
        "sensor_status": 1,
        "voltage": 230.0,
        "vfd_fault": 0,
        "plc_communication": 1,
        "machine_running": 1
    }
)

print("\nPLC DOCTOR - GEMINI TEST")
print("=" * 50)
print(result)
