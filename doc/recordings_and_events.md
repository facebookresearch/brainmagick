# Creating and handling recordings and events in brainmagick

This document provides an introduction to recordings and events in brainmagick.

## Glossary
* **Study**: set of M/EEG recordings produced as part of a scientific study. In brainmagick, each study must have its own module defined under `bm.studies`, which contains the code to download, load and format the raw data into `Recording` objects. The naming convention for studies is `<AuthorLastName><Year>`, e.g. `Gwilliams2022`.
* **Recording**: combination of the raw and/or preprocessed M/EEG data of a *single recording session* (stored as an mne.Raw object) along with the corresponding events (stored as a pandas DataFrame, see below). In brainmagick, recording objects inherit from `bm.studies.api::Recording` and follow the naming convention `<StudyName>Recording`, e.g. `Gwilliams2022Recording`.
* **Event**: external stimulus (e.g. word, audio clip, etc.) or internally generated action (e.g. press of a button) whose information is recorded alongside neuroimaging data during recordings. Events are used to create the targets of decoding analyses, e.g. predicting the word a participant was listening to based on their M/EEG data. Stored as a pandas `DataFrame` (see next item). Internally, events are represented by objects of the class `bm.events.Event`, with children classes for each event type (`Sound`, `Word`, `Phoneme`, etc.).
* **Event `DataFrame`**: a pandas `DataFrame` where each row contains information about one event (or its derivatives, i.e. a boundary or a block) which constitutes the primary interface for handling events in brainmagick. This DataFrame is built upon reading recordings for the first time and cached per-recording as a CSV file.
* **Block**: contains the (aggregated) event data found between two boundaries. Defined by a start time, a duration and a unique identifier number (UID) and stored along the events in the event `DataFrame`.

## Adding a new study
Adding new studies to brainmagick can be done with the following steps. Existing study modules are a good starting point for creating new studies.
1. Create new module under `bm/studies`. Follow the study name convention to name the new module, i.e. `<authorlastname><year>.py`.
2. Inside this new module, define a class called `StudyPaths`. This contains information about where to find the recordings and event information for the dataset.
3. Also define the recording class for your study, e.g. `<StudyName>Recording`. This class should implement methods `download`, `iter`, `_load_raw` and `_load_events`. For more information about how raw data and events are stored internally, see below.
4. Update the global study paths in `conf/study_paths/study_paths.yaml` to point to the directory where the data is stored.

## Raw M/EEG data
TODO

## Events

In brainmagick, events are stored as a pandas `DataFrame` where each row contains the information about a single event. Pandas `DataFrame`s are commonly used data structures in scientific Python as they represent and store tabular data (such as events) transparently and make it easy to access and modify its content.

The event `DataFrame` acts as the source of truth for events, and is used to cache and reload event-related information. Therefore, when creating a new dataset, it is necessary to ensure that events are properly formatted into a valid event `DataFrame`.

To simplify creating, validating and handling this `DataFrame`, brainmagick exposes a custom [pandas accessor](https://pandas.pydata.org/docs/development/extending.html#registering-custom-accessors). Brainmagick's `EventAccessor` gives access to a few useful methods that are described in more details in this section.

A quick example:
```python
import pandas as pd

events_df = pd.DataFrame(list_of_event_dicts)

# Make sure events are valid
events_df = events_df.event.validate()

# Create blocks from existing events
events_df = events_df.event.create_blocks(by='sentence')

# Plot events for sanity check
events_df.event.plot()
```

### Creating events

Each row of the event `DataFrame` contains the information about a single event. To this end, specific fields must be provided for each kind of event. To get a list of all required events per event kind, use

```python
events_df.event.list_required_fields()
```

Alternatively, this information can be found in the class definitions of `bm/events.py`.

For instance, all event kinds require the following fields:
* 'start'
* 'duration'
* 'kind'
* 'modality'
* 'language'

Additionally, specific event kinds require their own fields, e.g. `word` for word events, and `filepath` for sound events.

To ensure that the event `DataFrame` is valid, i.e. that each row contains the appropriate fields, use:

```python
events_df = events_df.event.validate()
```

If fields are missing or invalid values are provided, `validate()` will raise an error. For some event kinds, values might be transformed during validation - this is why it is important to update the event `DataFrame` with the output of the validation.

### Inspecting events

As the event `DataFrame` is a standard pandas `DataFrame`, it can be (pretty-)printed and manipulated as normally done with `DataFrame`s. To further facilitate the inspection of the events, e.g. for sanity checking when adding a new dataset or to validate the event design in a study recording, the `plot()` method is provided through the accessor:

```python
events_df.event.plot()
```

For instance, the following figure was obtained on part of the first recording of `Gwilliams2022`:
![Event visualization on first recording of Gwilliams2022](viz_example.png)

### Saving and loading

Event `DataFrame`s can be saved and loaded as CSV files using pandas's standard functionality:

```python
filename = '/somewhere/on/your/disk/events.csv'

# Save to filename
events_df.to_csv(filename)

# Load from filename
events_df = pd.read_csv(filename)

# Make sure the events are valid
events_df = events_df.validate()
```

### Adding new event types using the internal Event API

While the events are exposed through a standard pandas `DataFrame`, the definition of the event kinds can be found in a series of classes where each event kind is represented by a separate class. To add new event kinds, it is therefore necessary to create a new class that inherits from `bm.events::Event` to specify its required fields and implement any necessary checks or transforms of its input values.

First, create a new dataclass in `bm/events.py`:

```python
@dataclass
class MyNewEventKind(Event):
    field1: type
    field2: type
    ...

    def __post_init__(self):
        # Enforce valid values, e.g. field1 must be larger than 0
        assert self.field1 > 0
        # Transform values that need to be processed, e.g. apply a function on field2
        self.field2 = foo(self.field2)
```

Second, make sure to register the new event kind in the event accessor (`bm.events::EventAccessor`) by updating the `EVENT_KIND_MAPPING` dictionary:

```python
EVENT_KIND_MAPPING = {
    ...
    'myneweventkind': MyNewEventKind
}
```

Finally, new features (see `bm.features`) can be created to ingest the new kind of events.
