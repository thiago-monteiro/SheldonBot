<svelte:head>
	<title>SheldonBot</title>
	<meta name="robots" content="noindex nofollow" />
	<html lang="en" />
	<link
	rel="stylesheet"
	href="https://fonts.googleapis.com/icon?family=Material+Icons"
	/>
	<!-- Roboto -->
	<link
	rel="stylesheet"
	href="https://fonts.googleapis.com/css?family=Roboto:300,400,500,600,700"
	/>
	<!-- Roboto Mono -->
	<link
	rel="stylesheet"
	href="https://fonts.googleapis.com/css?family=Roboto+Mono"
	/>
	<link rel="stylesheet" href="/smui.css" media="(prefers-color-scheme: light)" />
	<link
	rel="stylesheet"
	href="/smui-dark.css"
	media="screen and (prefers-color-scheme: dark)"
	/>
</svelte:head>

<script>
	import "@accentdotai/recorderjs";
	import Button, { Label } from '@smui/button';

	URL = window.URL || window.webkitURL;

	var response = "";
	var ButtonText = "Record";
	var gumStream;
	var rec;
	var input;
	var AudioContext = window.AudioContext || window.webkitAudioContext;
	var audioContext

	function handleRecordButtonClick(){
		if (ButtonText == "Record"){
			startRecording();
			ButtonText = "Pause";
		}
		else if (ButtonText == "Pause"){
			stopRecording();
			ButtonText = "Record";
		}
	}

	function startRecording() {
		var constraints = { audio: true, video:false };

		navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
			audioContext = new AudioContext();
			gumStream = stream;
			input = audioContext.createMediaStreamSource(stream);
			rec = new Recorder(input, {numChannels:1})
			rec.record();
			console.log("Recording started");

		})
	}

	function stopRecording() {
        console.log("stopButton clicked");
        rec.stop();
        gumStream.getAudioTracks()[0].stop();
        rec.exportWAV(doPost);
    }

	function doPost(blob){
		var filename = new Date().toISOString();
		var xhr = new XMLHttpRequest();
		xhr.onload = function(e) {
			if (this.readyState === 4) {
				response = e.target.responseText;
			}
		};
		var fd = new FormData();
		fd.append("audio-data", blob, filename);
		xhr.open("POST", "http://localhost:5000/uploadAudio", true);
		xhr.send(fd);
	}
</script>
 
<div>
	<h1>Response: {response}</h1>
</div>
<div>
	<Button on:click={handleRecordButtonClick} variant="outlined"><Label> {ButtonText} </Label></Button>
</div>

<style>
	div {
		display: flex;
		justify-content: center;
		align-items: center;
	}
</style>