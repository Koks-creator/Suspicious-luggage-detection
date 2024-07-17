<h1>Suspicious luggage detection</h1>

<h2>Overview</h2>
<p>Goal of this project was to monitor any suspicious luggage, for example if luggage has been left for some time, and send alerts via telegram.</p>

[![video](https://img.youtube.com/vi/KCMV2qzWD0U/0.jpg)](https://www.youtube.com/watch?v=KCMV2qzWD0U)

<h2>Requirements</h2>
<ul>
  <li>Python 3.9+</li>
  <li>requirements.txt</li>
</ul>

<h2>How does it work</h2>
<p>To detect people and suitcases I've trained YoloV5 on data I've gathered, then suitcases are being tracked using SORT algorithm, after that program looks for onwers of detected suitcases, in order to do that it checks if any person's bbox crosses with suitcase's bbox. Margin of error is also being taken into account, because there are cases when bboxes don't cross, but suitcase is really close to potential owner, so you can add 10-20 pixels to suitcase bbox (<strong>ERROR_MARGIN in config.py</strong>). The next step is to switch between having owner and not (switching statuses - "Status" attribute in "data" dict), when bboxes cross, program add 1 (can be changed in config.py) to "FramesCount" attribute of "data" dict, "frame_count_thr" (can be changed in config.py) is a limit, so we avoid large "FramesCount", it's important since when bboxed don't cross we subtract 1 and once again "frame_count_thr" is a limit but in this case it's negative "frame_count_thr". This logic has been implented to avoid instant switches of statuses, instead it's better to make sure status can be changed and it's not some random and really short event. When it comes to states ("State" attribute), there are 3 of them: <strong>Warning, Suspicious, Very Suspicious</strong>, in config these states can be modified ("STATES" var). In "STATES" variable we can specify time threshold in seconds (as a key of dictionary), color and name. When suitcase has no owner, time counter is being triggered and state can be changed with respect to time counter, when some suitcase has status "Very Suspicious" telegram alert is being sent with information about id and area (if areas.pkl has been created - to create areas.pkl, take a frame of video you wanna use and run "space_picker.py"). To avoid spam, alerts are sent with pauses ("ALERTS_INTERVAL" in config.py).</p>

<h2>Related:</h2>
<ol>
  <li>https://github.com/Koks-creator/HowToTrainCustomYoloV5Model</li>
</ol>

<h2>Videos:</h2>
<ol>
  <li>https://www.pexels.com/pl-pl/video/pasazerowie-czekajacy-w-poczekalni-na-lotnisku-3723452/</li>
  <li>https://www.pexels.com/pl-pl/video/lot-swit-zachod-slonca-moda-8044791/</li>
  <li>https://www.pexels.com/pl-pl/video/osoba-kobieta-praca-pisanie-3695972/</li>
</ol>
