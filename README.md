# Event-Based Few-Shot Learning (FSL) for Human Activity Recognition

You can find the link to the website [here](https://mohammadbelalirshaid.github.io/HAR-FSL/)

## Project Overview
This project implements a novel pipeline for human activity classification using event-based data from a Dynamic Vision Sensor (DVS) and Few-Shot Learning (FSL) with Prototypical Networks. The system addresses catastrophic forgetting by preserving support prototypes and uses Euclidean distance for classification between support and query samples.

## Methodology
- **Data Acquisition**: DVS camera captures activities like walking, jumping, and boxing using ROS2.
- **Event Extraction & Conversion**: Event streams are converted into frames for analysis.
- **Few-Shot Learning Framework**: Prototypical Networks classify frames with minimal labeled data.
- **Mitigation of Catastrophic Forgetting**: Preserving prototype integrity throughout learning cycles.

## Results
### Accuracy Table
| Scenario        | Support Accuracy (%) | Query Accuracy (%) |
| --------------- | -------------------- | ------------------ |
| 5-shot Walking  | 57.68                | 0.00               |
| 5-shot Boxing   | 56.15                | 20.00              |
| 5-shot Jumping  | 55.94                | 40.00              |
| 10-shot Walking | 52.29                | 50.00              |
| 10-shot Boxing  | 50.54                | 30.00              |
| 10-shot Jumping | 63.50                | 70.00              |
| 20-shot Walking | 57.68                | 50.00              |
| 20-shot Boxing  | 68.95                | 80.00              |
| 20-shot Jumping | 49.17                | 95.00              |

## Figures
- **Boxing Recognition Example**
  
  ![Boxing1](https://github.com/user-attachments/assets/8bc394fa-1613-42e4-8d69-cde1673fb8b7)

- **Walking Recognition Example**
  
![Qualitative_training_testing_Jump_Box_Support - Copy](https://github.com/user-attachments/assets/80a90f74-9d40-4356-8801-e570f98ae0e8)

- **Jumping Recognition Example**
  
![Jumping](https://github.com/user-attachments/assets/db31477e-d86c-4255-9252-4c2b47d0ff24)

## Python Environment and Requirements
### Required Packages
Ensure the following Python packages are installed:
- `torch==2.0.0`
- `torchvision==0.15.1`
- `opencv-python==4.5.3`
- `Pillow==9.0.1`
- `matplotlib==3.6.0`
- `scikit-learn==1.2.0`

To install all dependencies:
```bash
pip install torch torchvision opencv-python Pillow matplotlib scikit-learn
