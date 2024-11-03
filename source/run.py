import subprocess

scripts = [
    'R1_BCG_classification.py',
    'R2a_backg.py',
    'R2b_make_input.py',
    'R3_richness_estimation.py',
    'R4_cal_r500.py',
    'R5_member_galaxy.py',
    'R6_scm_cm.py'
]

for script in scripts:
    print(f"Running {script} ...")
    try:
        result = subprocess.run(['python', script], capture_output=True, text=True, check=True)

        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        print(f"Script {script} failed, stopped.")
        break  

print("done.")
