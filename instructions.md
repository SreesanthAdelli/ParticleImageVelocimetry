# Interactive PIV Activity

Use a simulation to generate a vector field, record it as a video, and analyze it using Python!

## 1. Open the Vector Field

1. Go to: [https://editor.p5js.org/SihanZhang/sketches/Z-zdZZLzw](https://editor.p5js.org/SihanZhang/sketches/Z-zdZZLzw)
2. Press the **Play** button to start.
3. You can edit the code to change how the vector field behaves — try adjusting variables!

---

## 2. Record Your Vector Field

1. Clone the repository:

   ```bash
   git clone https://github.com/SreesanthAdelli/ParticleImageVelocimetry.git

2. Navigate to the cloned repository:

   ```bash
   cd ~/ParticleImageVelocimetry
   ```
3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Install SimpleScreenRecorder (Ubuntu only):

sudo apt update  
sudo apt install simplescreenrecorder

    If you are using a different operating system, use another screen recorder and save as an .mp4 file.

5. Launch SimpleScreenRecorder - enter simplescreenrecorder in the terminal or find it in your applications.

    simplescreenrecorder

    In the setup window:

        Choose "Record a fixed screen area" and manually select just the p5.js canvas.

        Set the frame rate to 2 FPS.

        Under Container, select mp4.

        Set the output file path to something like:
        ~/ParticleImageVelocimetry/yourname.mp4

    Click Start Recording and let your simulation run for at least 10 seconds.

    Click Save Recording to finish.

## 3. Run the PIV Analysis

Open a terminal.

Navigate to the cloned repository:

    cd ~/ParticleImageVelocimetry

Run the analysis:

    python run_piv_video.py yourname.mp4
