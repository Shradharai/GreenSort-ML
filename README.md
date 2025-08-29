# GreenSortML: Machine Learning-Based Energy-Aware GPU Sorting Framework

**Authors:**

* Kenisha Surana ([ks6295@srmist.edu.in](mailto:ks6295@srmist.edu.in))
* Akshita Sahu ([as3532@srmist.edu.in](mailto:as3532@srmist.edu.in))
* Shradha Rai ([sr1008@srmist.edu.in](mailto:sr1008@srmist.edu.in))
* Varenya Bhimaraju ([vb6341@srmist.edu.in](mailto:vb6341@srmist.edu.in))

**Affiliation:** Data Science and Business Systems, School of Computing, SRM Institute of Science and Technology, Kattankulathur, Chennai, India

---

## Overview

GreenSortML is a machine learning-based framework designed to optimize energy efficiency in GPU-accelerated sorting operations. Unlike traditional static approaches, GreenSortML dynamically selects the most energy-efficient GPU backend—**CUDA** or **OpenACC**—using predictive models based on input size, entropy, GPU temperature, and other runtime metrics.

The framework achieves significant energy savings without compromising runtime performance, making it suitable for energy-conscious computing environments such as green data centers, embedded GPUs, and edge nodes.

---

## Features

* **Energy-efficient backend selection:** Dynamically chooses CUDA or OpenACC for each sorting task.
* **Machine learning-driven:** Uses Gradient Boosting Regressors to predict energy consumption.
* **Real-time execution:** Backend selection latency is under 3 milliseconds.
* **Comprehensive profiling:** Monitors GPU temperature, bandwidth, and power consumption using NVML.
* **Supports multiple sorting algorithms:** Bitonic Sort, Radix Sort, Merge Sort.

---

## Technologies Used

* **Languages:** Python, C (for CUDA/OpenACC kernels)
* **Libraries & Frameworks:**

  * `scikit-learn` and `xgboost` for ML modeling
  * `pandas`, `numpy` for data processing
  * `matplotlib`, `seaborn` for visualization
  * `pynvml` for real-time GPU monitoring
* **GPU Frameworks:** CUDA 12.2, OpenACC (NVIDIA HPC SDK 23.7)

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/greensortml.git
```

2. Navigate to the project directory:

```bash
cd greensortml
```

3. Install Python dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset

* Synthetic dataset of 2,000 input arrays ranging from 1,000 to 1,000,000 elements.
* Entropy levels from 0.1 to 3.5 using normalized Shannon entropy.
* Features include: `size`, `entropy`, `sort_type`, `bandwidth`, `GPU temp`, `power_limit`.
* Target: Energy consumption (Joules) for CUDA and OpenACC backends.
* Data split: 60% training, 20% validation, 20% test.

---

## Model

* **Model Type:** Gradient Boosting Regressor (GBR)
* **Input Features:** `size`, `entropy`, `sort_type` (one-hot encoded), `bandwidth`, `temp`
* **Output:** Predicted energy consumption (Joules)
* **Transfer Learning:** Pretrained on synthetic data and fine-tuned on real GPU traces.

### Backend Selection Logic

```python
if EnergyOpenACC < EnergyCUDA and (RuntimeCUDA - RuntimeOpenACC)/RuntimeCUDA < 0.1:
    select OpenACC
else:
    select CUDA
```

* Ensures energy savings while maintaining runtime performance.

---

## Usage

```python
from greensortml import generate_dataset, GradientBoostingRegressor, predict_backend

# Generate synthetic dataset
df = generate_dataset()

# Train model
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# Predict best backend
backend = predict_backend(size=16384, entropy=0.5, sort_type='bitonic', bandwidth=512, temp=65)
print("Recommended backend:", backend)
```

---

## Evaluation Metrics

* **Energy Savings:** 12.1% on average compared to CUDA-only baseline.
* **Backend Selection Accuracy:** 88.5%.
* **Prediction Overhead:** \~2.84 ms per selection.
* **Runtime Impact:** <2.7% deviation from fastest backend.
* **ROC AUC for backend selection:** 0.921

---

## Visualization

* Scatterplots of energy vs. input size and entropy.
* Bar plots comparing CUDA, OpenACC, and ML hybrid models.
* ROC curves for backend selection accuracy.

Example:

```python
import seaborn as sns
import matplotlib.pyplot as plt
sns.scatterplot(data=df, x='size', y='joules', hue='entropy', style='type')
plt.show()
```

---

## Conclusion

GreenSortML demonstrates a robust approach to energy-efficient GPU sorting through machine learning-based backend selection. It combines predictive modeling with runtime profiling to save energy without significant runtime overhead, making it ideal for modern energy-conscious computing environments.

---

## References

1. NVIDIA. CUDA Toolkit Documentation, 2024.
2. OpenACC Programming and Best Practices Guide, 2024.
3. NVIDIA Management Library (NVML) Developer Guide, 2024.
4. Paszke et al., PyTorch: An Imperative Style, High-Performance Deep Learning Library, NeurIPS 2019.
5. Pedregosa et al., Scikit-learn: Machine Learning in Python, JMLR 2011.
6. Matplotlib & Seaborn Documentation.
7. Xu, J., Wang, L., et al. An Energy-Aware GPU-Accelerated Sorting Framework, ICCGC 2022.

---

## License

This project is licensed under the MIT License.
censed under the MIT License. See the [LICENSE](LICENSE) file for details.
# GreenSort-ML
