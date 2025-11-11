from datetime import datetime, timedelta
from timeseries import *
import json
import os


INPUT_DIR = os.getenv('INPUT_DIR')
OUTPUT_DIR = os.getenv('OUTPUT_DIR')
HEADER_BYTES = 256 
EVENT_LABEL_BYTES = 5632
FREE_SPACE_BYTES = 1024
ANNOTATION_SIZE = 6
EVENT_BLOCKS = [(16,16), (16,16), (32,64), (64,64)] # (bit length, num rows)
event_labels = []
TVX_TERMINATOR = 0x00000000

def getInputFiles():
    """
    Get the input files from the INPUT_DIR environment variable.
    Expects exactly one .tvx file
    Raises ValueError if the number of files is not as expected.
    """
    tvx_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith('.tvx')]

    if len(tvx_files) != 1:
        raise ValueError(f"Expected exactly one .tvx file in {INPUT_DIR}, found {len(tvx_files)}")

    input_tvx = os.path.join(INPUT_DIR, tvx_files[0])
    return input_tvx

def readFile(input_tvx):
    """
    Read .tvx file
    """
    raw_data = ""
    with open(input_tvx, 'rb') as file:
        raw_data = file.read()
    
    return raw_data

def findAnnotationsBlocks(raw_data):
    """
    Expect annotations to come in blocks of 6 bytes.
    A pattern of TVX_TERMINATOR (0x00000000) will be found at the end of the annotations.  
    If the terminator is not found, use the rest of the file.
    """
    annotation_hex = raw_data[(HEADER_BYTES + EVENT_LABEL_BYTES + FREE_SPACE_BYTES):]
    terminator_bytes = TVX_TERMINATOR.to_bytes(4, byteorder='big')
    annotations_end = annotation_hex.find(terminator_bytes)
    if annotations_end == -1:
        # Terminator not found, use entire remaining data
        return annotation_hex
    
    annotation_hex = annotation_hex[:annotations_end]
    return annotation_hex

def readHeader(raw_data):
    """
    Read in 256 bytes(512 chars) of header data from the raw data.
    """
    VERSION = (0,8)
    STUDY_CODE = (88,100)
    DATE = (168,176)
    BYTES_BEFORE_TIMING = (184,192)

    version = raw_data[VERSION[0]:VERSION[1]].decode('utf-8').strip()
    study_code = raw_data[STUDY_CODE[0]:STUDY_CODE[1]].decode('utf-8').strip()
    date = raw_data[DATE[0]:DATE[1]].decode('utf-8').strip()
    bytes_before_timing = raw_data[BYTES_BEFORE_TIMING[0]:BYTES_BEFORE_TIMING[1]].decode('utf-8').strip()

    header_info = {
        'version': version,
        'study_code': study_code,
        'date': date,
        'header_event_label_bytes': int(bytes_before_timing)
    }
    return header_info


def getEventLabels(raw_data):
    """
    Read in event labels
    Event labels are the next 5632 bytes after the header
    The first group of events are 16 entries x 16 bytes
    The second group of events are 16 entries x 16 bytes
    The third group of events are 32 entries x 32 bytes
    The final group of events are 64 entries x 64 bytes
    """
    raw_events = raw_data[HEADER_BYTES:EVENT_LABEL_BYTES]
    labels = []

    event_cursor = 0
    for events in EVENT_BLOCKS:
        for _ in range(events[1]): 
            start_range = event_cursor
            end_range = (start_range+events[0])
            annotation_substring_hex = raw_events[start_range:end_range]     

            read_event = annotation_substring_hex.decode('utf-8').strip()
            labels.append(read_event)
            event_cursor += events[0]

    return labels

def processEvent(eventHex):
    """
    Events are in 3 words, 16 bits each for a total of 6 bytes
    The least significant byte comes first in the file for all words
    The first word contains the Event Number and a general-purpose 
        numeric values associated with the event type: for Photic stimulation 
        it 1s the rate in Hz, for Hyperventilation it 1s the elapsed time in multiples of 10s
        Follows the form [Ev7, Ev6, Ev5, Ev4, Ev3, Ev2, Ev1, Ev0, N7, N6, N5, N4, N3, N2, N1, N0]
    The second word gives the Month(1-12), Day(1-31) and Hour(0-23) detail
        [M3, M2, M1, M0, D4, D3, D2, D1, D0, H4, H3, H2, H1, H0, X, X]
        The last two bits are unused. The year of the recording is the same the header
    The third word gives the number of Minutes, Seconds and the Fractional mantissa part of
        the nubmer of seconds in binary 1/16th seconds 0-15
        [M5, M4, M3, M2, M1, M0, S5, S4, S3, S2, S1, S0, FM3, FM2, FM1, FM0]
    """

    annotations = chunkAnnotations(eventHex,ANNOTATION_SIZE)
    annotation_details = []
    for annotation in annotations:
        events = {}
        little_endian_annotation = convertToLittleEndian(annotation)
        events.update(getEventPhoticSimulation(little_endian_annotation[0]))
        dateTimeHeader =parseDatetimeHeader(little_endian_annotation[1])
        events.update(dateTimeHeader)
        dateTimeDetails= parseTimeDetails(little_endian_annotation[2])
        events.update(dateTimeDetails)
        annotation_details.append(events)
        print(events)
    
    return annotation_details

    

def chunkAnnotations(string, length):
    groups = [string[i:i+length] for i in range(0, len(string), length)]
    return [[chunk[j:j+2] for j in range(0, len(chunk), 2)] for chunk in groups]

def convertToLittleEndian(hex_words):
    """
    Flip each 16-bit word in a list from big-endian to little-endian.
    E.g., ['0002', '2c43', 'a88c'] → ['0200', '432c', '8ca8']
    """
    little_endian = []
   
    for word in hex_words:
        # Ensure word is bytes, not str
        if isinstance(word, str):
            word_bytes = bytes.fromhex(word)
        else:
            word_bytes = word
        if len(word_bytes) != 2:
            raise ValueError(f"Expected 2 bytes per word, got {len(word_bytes)} bytes in '{word}'")
        # Swap the two bytes
        little_bytes = word_bytes[::-1]
        little_endian.append(little_bytes.hex()) 

    return little_endian

def hexToBinary(hex_string):
    value = int(hex_string, 16)
    binary_16bit = format(value, '016b')

    return binary_16bit

def getEventPhoticSimulation(word):
    binary = hexToBinary(word)
    event = int(binary[0:8], 2)
    photicSimulation = int(binary[8:], 2)
    return {
        'event': event,
        'photicSimulation': photicSimulation
    }

    
    pass

def parseDatetimeHeader(word):
    binary = hexToBinary(word)
    month = int(binary[0:4], 2)
    day = int(binary[4:9], 2)
    hour = int(binary[9:14], 2)
    return {
        'month': month,
        'day': day,
        'hour': hour
    }

def parseTimeDetails(word):
    binary = hexToBinary(word)
    minute = int(binary[0:6], 2)
    seconds = int(binary[6:12], 2)
    fractional_seconds = int(binary[12:], 2)
        
    return {
        'minute': minute,
        'seconds': seconds,
        'fractional_seconds': fractional_seconds
    }

def buildJSON(event_labels, events,date):
    pass

def buildJson(events, event_labels, date):
    tvx_datetime = datetime.strptime(date, "%d.%m.%y")
    result = []
    prev_event_time = 0
    base_start_time = 0
    MICROSECONDS_PER_16TH_SECOND = int(1_000_000 / 16)  # 62_500

    for idx, event in enumerate(events):
        event_time = tvx_datetime.replace(
            hour=event['hour'],
            minute=event['minute'],
            second=event['seconds'],
            microsecond = int(event['fractional_seconds']) * MICROSECONDS_PER_16TH_SECOND
        )

        start_epoch = int(event_time.timestamp() * 1000)

        if idx == 0:
            base_start_time = event_time
            relative_start = 0
        else:
            prev_event_time = tvx_datetime.replace(
                hour=events[idx - 1]['hour'],
                minute=events[idx - 1]['minute'],
                second=events[idx - 1]['seconds'],
                microsecond=int((events[idx - 1]['fractional_seconds'] / 16) * 1e6)
            )
            
            relative_start = int((event_time - base_start_time).total_seconds() * 1000)
        label_index = event['event'] if event['event'] < len(event_labels) else 0
        label = event_labels[label_index]

        result.append({
            "name": label,
            "label": label,
            "start": start_epoch,
            "end": start_epoch,
            "relative_start": relative_start * 1000,
            "relative_end": relative_start * 1000
        })
    if result:
        result[-1]["relative_start"] -= 3_000_000
        result[-1]["relative_end"] -= 3_000_000

    return result

def extract_annotations():
    input_tvx = getInputFiles()
    raw_data = readFile(input_tvx)
    header = readHeader(raw_data)
    event_labels = getEventLabels(raw_data)
    annotations_block = findAnnotationsBlocks(raw_data)
    events = processEvent(annotations_block)
    annotations = buildJson(events, event_labels, header['date'])
    
    
    with open(f"{OUTPUT_DIR}/annotations.json", "w") as f:
        json.dump(annotations, f, indent=2)
        print(f"wrote out CSV to {f.name}")
    
    session_key = authenticate()
    workflowData = getWorkflowData(session_key)


    datasetId = workflowData['datasetId']
    print(f"Dataset ID: {datasetId}")
    bdfPackage = getBDFPackageId(session_key, datasetId)
    print(f"BDF Package ID: {bdfPackage['package_id']}")
    annotationLayer = createAnnotationLayer(session_key, bdfPackage['package_id'])
    print(f"Annotation Layer Created: {annotationLayer}")
    channels = getChannels(session_key, annotationLayer['timeSeriesId'])
    print(f"Channels: {channels}")
    timeseriesIdPackageName = annotationLayer['timeSeriesId']
    print(f"Timeseries ID Package Name: {timeseriesIdPackageName}")
    timeseriesId =  annotationLayer['id']
    print(f"Timeseries ID: {timeseriesId}")

    createAnnotation(session_key,channels,timeseriesId,timeseriesIdPackageName,annotations)

