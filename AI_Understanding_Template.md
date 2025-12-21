# Reference Notes
1) Get code and comment from *.ipynb files
    jupyter nbconvert --no-prompt --to script 03_classification.ipynb
    # change name of file as required
2) Required data from python code:
    <details>
        <summary><b> ✅AI Understanding Template-ch0_XXX_01</b>
        </summary>
                    
    </details>


    Please provide answer for AI w.r.t. code explanation: 
    • What is it? 
    • How does it reason? 
    • Where does it fail? 
    • When should I use it? 
    • What is the mental model? 
    • How do I prompt it? 
    • What are alternatives? 

    • Explain the code, get key terms and provide their full definition as well
    • cheatsheet
    • 1 page MCQ with answer
    • Scenario-Based Interview Questions 

    <details>
        <summary><b> ✅AI Understanding Template-ch0_01_Please provide answer for AI w.r.t. code explanation:</b>
        </summary>
                    
    </details>

    <details>
        <summary><b> ✅AI Understanding Template-ch15_01_Explain the code, get key terms and provide their full definition as well</b>
        </summary>
                    
    </details>

    <details>
        <summary><b> ✅AI Understanding Template-ch0_01_cheatsheet</b>
        </summary>
                    
    </details>

    <details>
        <summary><b> ✅AI Understanding Template-ch0_01_1 page MCQ with answer</b>
        </summary>
                    
    </details>

    <details>
        <summary><b> ✅AI Understanding Template-ch0_01_Scenario-Based Interview Questions</b>
        </summary>
                    
    </details>


3) Python
    Here’s the **short, clear answer** you prefer 👇

---

# ✅ **In Python, what is a module / package / library? And is it connected to the system?**

## 🔹 **MODULE**

A single `.py` file.
Example:
`math.py` → a module.

### Where it comes from?

* Built-in with Python (e.g., `math`, `sys`, `os`)
* Or created by you

### System connection?

* Only built-in modules like `os`, `sys`, `subprocess` can interact with the system.
* Most modules do NOT touch your system.

---

## 🔹 **PACKAGE**

A folder containing multiple modules + an `__init__.py` file.
Example structure:

```
mypkg/
    __init__.py
    utils.py
    data.py
```

---

## 🔹 **LIBRARY**

A collection of **packages + modules** providing functionality.
Example:
`numpy`, `pandas`, `matplotlib` → libraries

A library **may contain** many packages and modules.

---

# 🧩 **Are they connected to the system?**

## 1️⃣ **Built-in standard libraries**

These come with Python and are safe.

Some interact with the system:

* `os` → filesystem, environment variables
* `sys` → Python interpreter info
* `subprocess` → run system commands
* `pathlib` → file paths

Others are pure computation:

* `math`
* `statistics`
* `json`

---

## 2️⃣ **Installed libraries (pip installs)**

Installed under your environment:

```
site-packages/
```

These are **NOT connected to the OS** unless they are designed to be.
Example:

* `numpy` → no system access
* `pandas` → no system access
* `sklearn` → no system access
* `requests` → access to network (not OS)

---

## 3️⃣ **Can Python modules harm the system?**

Only **if**:

* you use modules like `os.remove()`, `shutil.rmtree()`
* you run untrusted packages containing harmful code

Otherwise… **no**.

---

# ⭐ **Super simple summary**

| Thing       | Meaning               | System Access?               |
| ----------- | --------------------- | ---------------------------- |
| **Module**  | One file              | Only if it's a system module |
| **Package** | Folder of modules     | Same as above                |
| **Library** | Many packages/modules | Same as above                |

---

If you want, I can also give you:

✅ Diagram: module vs package vs library
✅ Example code for each
✅ How Python searches for modules (`sys.path`)
Just tell me!

4) 