tuflow-flash
==========================================

Program to start an integrated Tuflow flash simulation including data collection, pre- and postprocessing.


Installation and usage
----------------------

Preferably set up an anaconda environment for the simulations with::
	$ conda create -n <environment name> python=3.8 anaconda
	
The installation requires GDAL to be pre-installed. Install Gdal with::
	$ conda install -c conda-forge gdal

Adapter itself can be installed using python 3.6+ with::

  $ pip install git+https://github.com/lokhorstivar/tuflowflash.git

The script is called ``run-tuflow-flash``, you can pass ``--help`` to get usage
instructions and ``--verbose`` to get more verbose output in case of
problems.

``run-tuflow-flash`` looks for a ``settings.ini`` in the current directory by
default, but you can pass a different file in a different location with
``--settings``::

  $ run-tuflow-flash
  $ run-fews-3di --help
  $ run-fews-3di --settings /some/directory/settings.ini


Configuration and input/output files
------------------------------------

The expected information in settings.ini is::

  [general]
  relative_start_time=-3
  relative_end_time=0
  
  [tuflow]
  tuflow_executable=C:\Users\922383\Downloads\TUFLOW.2020-10-AD\2020-10-AD\TUFLOW_iDP_w64.exe
  tcf_file=M01_5m_002.tcf
  prepare_state_for_next_run=True
  gauge_rainfall_file=C:\Users\922383\Documents\TUFLOW\bc_dbase\100yr2hr_rf.csv
  netcdf_rainfall_file=C:\Users\922383\Documents\TUFLOW\bc_dbase\RFG\test.nc
  
  [lizard]
  apikey=<yoursecretkey>
  precipitation_uuid_file=precipitation_gauges.csv
  depth_raster_uuid=cbb5a24a-9e42-434b-ae29-255d6e96c416
  rainfall_raster_uuid=95b3b6cb-af04-41b4-ad56-a356faab8217
  
  [bom]
  bom_file=IDR318EN.RF3
  bom_url=ftp://ftp.bom.gov.au/anon/sample/catalogue/Radar/Rainfields/
  
  [switches]
  get_historical_precipitation=True
  get_future_precipitation=True
  run_simulation=True
  post_to_lizard=True	
