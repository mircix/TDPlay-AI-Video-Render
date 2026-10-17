import sys

if sys.argv[1:2] == ["download"]:
    from tdplay.models import cli_download

    cli_download(sys.argv[2:])
else:
    from tdplay.app import main

    main()
