import json
from shutil import copyfile
import subprocess
import os
from tqdm import tqdm

class AeneasAligner():
	def __init__():
		import aeneas
		from aeneas.executetask import ExecuteTask
		from aeneas.tools.execute_task import ExecuteTaskCLI
		from aeneas.task import Task
		from aeneas.job import Job


	def align(self, text, audiopath, tokenized_text_output_path="tmpfile", sync_map_file_path="tmpfile", removeoutputs=True, lang="eng", tokenizer= lambda x : x.split()):
		"""
	    Align a sentence to an audio file at the word level (by default). If you would like a different level (bigrams/phrases/subword) etc, change the tokenizer as you wish.

	    Parameters
	    ----------
	    text : str
	        A string, the text to be aligned
	    audiopath : str
	        The path to the audiofile to be aligned
		tokenized_text_output_path : str, optional
			The path to save the intermediate tokenized inputs before passing them to aeneas
		sync_map_file_path : str, optional
			The path to save the output alignment maps from aeneas
		removeoutputs : bool, default: True
			Whether or not to remove the intermediate file inputs and outputs
		lang : str, default: eng
			The code of the language of the text. This will be used for the simple TTS algorithm before applying Dynamic Time Warping
		tokenizer : a function
			A function, which when given a sentence returns an array of tokens. default: x->x.split(" "), which splits based on spaces 


	    Returns
	    -------
	    An array of dicts, where each dict is of the form:
	    	{
			"token": the token in question (could be a bigram/sentence/phrase/subword depending on your tokenizer)
			"begin": the start position of the token in the audio, in seconds. eg: 4.32 means starting at 00:00:04:320
			"begin": the end position of the token in the audio, in seconds.
	    	}

	    Examples
	    --------
	    >>> aligner.align("Hello world", "myaudiofile.flac")
	    [
			{
			"token": "Hello",
			"begin": 0.0,
			"end": 0.480
			}, 
			{
			"token": "world",
			"begin": 0.480,
			"end": 0.760
			}
	    ]
	    """
		config_string = u"task_language={}|is_text_type=plain|os_task_file_format=json".format(lang)
		task = Task(config_string=config_string)
		task.audio_file_path_absolute = audiopath
		# task.text_file_path_absolute = u"/path/to/input/plain.txt"
		taskfilename = os.path.join(os.getcwd(),f"{tokenized_text_output_path}.txt")

		tokens = tokenizer(text)

		# Create text file
		with open(taskfilename, "w") as fl:
			for token in tokens:
				print(token, file=fl)

		task.text_file_path_absolute = taskfilename
		task.sync_map_file_path = u"{}.json".format(sync_map_file_path)

		ExecuteTask(task).execute()
		task.output_sync_map_file("")

		sync_map = json.load(open(task.sync_map_file_path))

		result = []

		print(sync_map["fragments"])

		for token, timestamp in zip(tokens, sync_map["fragments"]):
			print(timestamp)
			result.append({
				"token":token,
				"begin":float(timestamp["begin"]),
				"end":float(timestamp["end"])
				})

		if removeoutputs:
			os.remove(task.sync_map_file_path)
			os.remove(task.text_file_path_absolute)

		return result


	def align_batch(self, texts, audiopaths, workdir="workdir", outputdir="outputs", lang="eng", tokenizer= lambda x : x.split(), getfileprefix=lambda x:""):
		"""
	    Aligns a batch of sentences to audio files at the word level (by default). If you would like a different level (bigrams/phrases/subword) etc, change the tokenizer as you wish.

	    Parameters
	    ----------
	    text : str
	        A string, the text to be aligned
	    audiopath : str
	        The path to the audiofile to be aligned
		workdir : str, optional
			The path to store the intermediate inputs before passing them to aeneas
		outputdir : str, optional
			The path to save the output alignment maps from aeneas
		lang : str, default: en
			The ISO code of the language of the text. This will be used for the simple TTS algorithm before applying Dynamic Time Warping
		tokenizer : a function
			A function, which when given a sentence returns an array of tokens. default: x->x.split(" "), which splits based on spaces 
		getfileprefix : a function
			A function, which when given the path of the audio, returns the desired prefix of the path of the alignment map. The alignments will be saved to /outputdir/prefix/map.json
			Default, a function which returns no prefix. This essentially puts all alignment maps into the main output dir.

	    Returns
	    -------
		Nothing. The output sync maps are saved to the output dir as a zip file for further processing if needed.

	    Examples
	    --------
	    >>> aligner.align_batch(["Hello world", "No, thank you", "Hello there"], ["myaudiofile1.flac", "myaudiofile2.flac", "myaudiofile3.flac"])
	    """
		assert not texts is None, "texts cannot be None" 
		assert not audiopaths is None, "audiopaths cannot be None" 
		assert len(audiopaths) > 0, "audiopaths is of length <= 0"


		audio_extension = os.path.splitext(audiopaths[0])[-1]

		config = f"""is_hierarchy_type=flat
is_hierarchy_prefix=inputs/
is_text_file_relative_path=.
is_text_file_name_regex=.*\\.txt
is_text_type=plain
is_audio_file_relative_path=.
is_audio_file_name_regex=.*\\{audio_extension}

os_job_file_name=batchout
os_job_file_container=zip
os_job_file_hierarchy_type=flat
os_job_file_hierarchy_prefix=outputs/
os_task_file_name=$PREFIX.json
os_task_file_format=json

job_language={lang}
job_description=AENEAS batch job
"""

		os.makedirs(workdir, exist_ok=True)
		os.makedirs(os.path.join(workdir, "inputs"), exist_ok=True)
		os.makedirs(outputdir, exist_ok=True)

		# Write config file
		with open(os.path.join(workdir, "config.txt"), "w") as configfile:
			configfile.write(config)

		for text, audiopath in zip(texts, audiopaths):
			# Get file prefix, if function provided
			fileprefix = getfileprefix(audiopath)
			# Extract audio file name
			audiofilename = os.path.basename(audiopath)
			textfilename = os.path.splitext(audiofilename)[0] + ".txt"

			# Copy to job path
			copyfile(audiopath, os.path.join(workdir, "inputs", fileprefix, audiofilename))

			# Create tokenized input text file
			tokens = tokenizer(text)
			with open(os.path.join(workdir, "inputs", fileprefix, textfilename), "w") as txtfl:
				for token in tokens:
					print(token, file=txtfl)

		print("Executing: {}".format(f"python -m aeneas.tools.execute_job {workdir} {outputdir}"))
		subprocess.call(f"python -m aeneas.tools.execute_job {workdir} {outputdir}", shell=True)

		# ecli.run(
		# 	arguments = ["", workdir+"/", outputdir]
		# 	)

class MfaAligner():

	def align(self, text, audiopath, tokenized_text_output_path="tmpfile", removeoutputs=True, lang="eng", tokenizer= lambda x : x.split()):
	    raise NotImplementedError("Cannot align 1 instance using MFA, please use batched alignment or the AENEAS aligner.")
		

	def align_batch(self, texts, audiopaths, workdir="workdir", lang="english", tokenizer= lambda x : x.split(), getfileprefix=lambda x:"", prefixes=None, overwrite=False, skipprep=False, validate=False, numjobs=2, cleanup=True):
		"""
	    Aligns a batch of sentences to audio files at the word level using the montreal forced aligner.

	    Parameters
	    ----------
	    text : str
	        A string, the text to be aligned
	    audiopath : str
	        The path to the audiofile to be aligned
		workdir : str, optional
			The path to store the intermediate inputs before passing them to MFA
		lang : str, default: english
			The ISO code of the language of the text. This will be used for the simple TTS algorithm before applying Dynamic Time Warping
		tokenizer : a function
			A function, which when given a sentence returns an array of tokens. default: x->x.split(" "), which splits based on spaces 
		getfileprefix : a function
			A function, which when given the path of the audio, returns the desired prefix of the path of the alignment map. The alignments will be saved to /outputdir/prefix/map.json
			Default, a function which returns no prefix. This essentially puts all alignment maps into the main output dir.  
		prefixes : an array of size equal to audiopaths and texts, with the prefix to put each audiofile at. Takes precedence over getfileprefix. default: None
		skipprep: boolean, whether to skip data preparation. Useful for restarting failed runs. Default: False
		overwrite : boolean, whether to overwrite existing files. If false, working directory must be empty. Default: False 
		validate : boolean, whether to validate files for MFA. Warning: Takes a very long time and increases processing time considerably. Default: False 
		numjobs : number of threads to use, default: 2
		cleanup : boolean, whether to clean up the intermediate files. Default: True 

	    Returns
	    -------
		Nothing. The output sync maps are saved to the output dir as a zip file for further processing if needed.

	    Examples
	    --------
	    >>> aligner.align_batch(["Hello world", "No, thank you", "Hello there"], ["myaudiofile1.flac", "myaudiofile2.flac", "myaudiofile3.flac"])
	    """
		assert not texts is None, "texts cannot be None" 
		assert not audiopaths is None, "audiopaths cannot be None" 
		assert len(texts)==len(audiopaths), "audiopaths must be of same length as texts"
		assert prefixes is None or len(prefixes) == len(texts), "prefixes must be None or of same length as texts and audiopaths"
		assert len(audiopaths) > 0, "audiopaths is of length <= 0"

		inputdir = os.path.join(workdir, "inputs")
		outputdir = os.path.join(workdir, "outputs")
		tmpdir = os.path.join(workdir, "tmp")

		os.makedirs(workdir, exist_ok=True)

		if not overwrite:
			assert len(os.listdir(workdir)) == 0, "overwrite set to false, working directory not empty. Please empty the working directory or set overwrite to true."

		os.makedirs(inputdir, exist_ok=True)
		os.makedirs(outputdir, exist_ok=True)
		os.makedirs(tmpdir, exist_ok=True)

		if not skipprep:
			for idx, (text, audiopath) in tqdm(enumerate(zip(texts, audiopaths)), desc="Preparing data"):
				if prefixes is not None:
					fileprefix = prefixes[idx]
				else:
					# Get file prefix, if function provided
					fileprefix = getfileprefix(audiopath)
				# Extract audio file name
				audiofilename = os.path.basename(audiopath)
				textfilename = os.path.splitext(audiofilename)[0] + ".lab"

				if not os.path.exists(os.path.join(inputdir, fileprefix)):
					os.makedirs(os.path.join(inputdir, fileprefix))

				# Symbolic links do not work with mfa
				# if symbolic_link: 
				# 	os.symlink(audiopath, os.path.join(inputdir, fileprefix, audiofilename))
				# else:
				
				# # Copy to job path
				copyfile(audiopath, os.path.join(inputdir, fileprefix, audiofilename))

				# Create tokenized input text file
				tokens = tokenizer(text)
				with open(os.path.join(inputdir, fileprefix, textfilename), "w") as txtfl:
					print(" ".join(tokens), file=txtfl, end="")

		if validate:
			print("Executing: {}".format(f"mfa validate {inputdir} {lang} {lang} -j {numjobs} --clean -t {tmpdir}" + (" --overwrite" if overwrite else "")))
			subprocess.call(f"mfa validate {inputdir} {lang} {lang} -j {numjobs} --clean -t {tmpdir}" + (" --overwrite" if overwrite else ""), shell=True)
		else:
			print("Skipping validation")
		print("Executing: {}".format(f"mfa align {inputdir} {lang} {lang} {outputdir} -j {numjobs} --clean -t {tmpdir} --disable_mp" + (" --overwrite" if overwrite else "")))
		subprocess.call(f"mfa align {inputdir} {lang} {lang} {outputdir} -j {numjobs} --clean -t {tmpdir} --disable_mp" + (" --overwrite" if overwrite else ""), shell=True)

		if cleanup and len(os.listdir(outputdir))!=0:
			subprocess.call(f"rm {inputdir}/* -rf", shell=True)
			subprocess.call(f"rm {tmpdir}/* -rf", shell=True)

		return len(os.listdir(outputdir)) != 0