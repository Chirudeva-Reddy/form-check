# Form Check

**Try it live:** https://chirudeva-reddy.github.io/form-check/

Prop up your phone, pick an exercise and lift. The page counts your reps, draws your skeleton and speaks short form cues after each rep. A dashboard breaks down each rep: the joint-angle curve, end range, range of motion, lowering and lifting time, lift speed against your best rep, and torso lean or the left/right gap. You can also analyze a video you already recorded.

Everything runs in your browser. The camera feed and your videos never leave your device; the page only downloads the pose model and the MediaPipe library.

> **Beta.** Feedback is coaching, not medical advice. Every threshold is an educated guess that hasn't been checked against labelled clips yet.

## Exercises

| Exercise | Camera | What it checks |
|---|---|---|
| Squat | side-on | depth (hips to knee level), chest up, tempo |
| Lunge | side-on | depth, upright torso, tempo |
| Deadlift | side-on | lockout at the top, hips too low at the start (squatting the bar up), tempo. Start with the bar on the floor; each rep counts when the bar is back down |
| Romanian deadlift | side-on | hinge depth, knees soft but still, standing tall between reps, tempo |
| Bent-over row | side-on | full pull and full stretch, torso hinged forward, torso kept still, tempo |
| Push-up | side-on, camera low | depth, lockout, hips sagging or piking, tempo |
| Bicep curl | facing or side-on | full range at the top and bottom, elbows by your sides, no swinging (side view), tempo |
| Shoulder press | facing or side-on | lockout overhead, back down to shoulder height, both arms even (facing), no leaning back (side view), tempo |
| Lateral raise | facing | up to shoulder height but not above it, both arms even, tempo |

Film from 2–3 m away with your whole body in frame. If the camera can't see what it needs, the page says so ("Turn side-on", "I can't see your feet") instead of guessing.

## How it works

1. **Pose:** [MediaPipe Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) (Full model; add `?lite` to the URL for older phones) finds 33 body points in every frame, in the browser.
2. **Pick the lifter:** it starts on the biggest, most central person, then stays on them. A passer-by can't take over tracking, and a warning shows while someone else is in view.
3. **Reject glitches:** if a joint jumps further than a body can move in one frame, the frame is skipped. Three such frames in a row mean the movement was real, and tracking resets.
4. **Smooth:** a [One-Euro filter](https://gery.casiez.net/1euro/) removes jitter without lagging real movement.
5. **Gate:** it checks the camera angle (from shoulder width compared with torso length) and joint visibility, and doesn't count frames it can't trust. Side-on lifts keep reading the same side of your body unless the other side becomes clearly more visible. Timing is measured in milliseconds, so a 15 fps upload and a 30 fps camera behave the same.
6. **Count:** each exercise tracks one joint angle (knee, elbow or shoulder). A rep runs from the start position to the far end and back. Two thresholds, an earlier one to start a rep and a later one to finish it, stop jitter from double-counting. The start position calibrates itself to your body from the first frames.
7. **Judge:** fixed rules on the finished rep, for example "hip at or below knee height at the bottom". There's no AI guessing; every cue comes from a measured angle or position.
8. **Measure:** each rep also gets timing, speed and symmetry numbers for the dashboard. The page explains each one and links the studies behind it. Lift speed is a joint-angle speed, not bar speed, so it only means something compared with your own reps.

All the logic lives in [`engine.js`](engine.js). Adding an exercise means adding one recipe: the joints it needs, the camera views it allows, the tracked angle and its start and end values, and a list of checks.

## Develop

No build step and no dependencies.

```bash
node --test                    # engine tests on synthetic stick figures (Node 20+)
python3 -m http.server 8765    # then open http://localhost:8765
```

On a phone the camera needs HTTPS, so test live mode on the deployed GitHub Pages site.

## Credits

The idea of turning pose estimation into rep counting and per-rep feedback came from [jeremyipark/vision-demos](https://github.com/jeremyipark/vision-demos), which runs offline Python pipelines with cloud models. This project shares no code with it. Pose detection uses [MediaPipe](https://github.com/google-ai-edge/mediapipe) (Apache-2.0).

## License

[MIT](LICENSE)
