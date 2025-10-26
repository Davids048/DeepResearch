# 2025-10-26 21:58:15  
- Implemented phase 1 of iterative evolution. 
    - add a for loop with max_iteration. 
    - still returning the same things into run_multi_react.py, but with some additions. 
- known issues: 
    - [ ] reflector is flawed: when agent concluded that the question is wrong (e.g. ), the reflector 
    agrees, and marks the agent's prediction as correct. ==> Might need to update the reflector's prompt. 
    - [ ] The agent currently has know recognition of previous trials. So it tries the same thing. (no progress/exploration).


