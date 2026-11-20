"""The glovebox flake scanner (Olympus BX53M): one window to say which flakes are wanted (thickness, size, cleanliness),
one scan (10x, analysis, 100x), and the 100x pictures sorted into wanted flakes, other real flakes and glue.

    tools\\bx53m\\run_scanner.bat            (python -m flakeid.gscan)

It is a second program beside the original one (python -m flakeid.scope, tools\\bx53m\\run_flakeid.bat), which is not
replaced: the stage, focus, camera and autofocus code, and the focus safety limits, are the original program's
(flakeid.scope.scan.Scanner), started here with the settings the wanted flakes need.

    criteria.py   what is wanted, and the scan settings that follow from it
    app.py        the window
    engine.py     one scan job through the original program's Scanner (its own process: python -m flakeid.gscan.engine)
    collect.py    after the scan: every 100x picture joined to the 10x region under its middle, judged real or glue,
                  its thickness checked against the 100x colour, sorted by the criteria
    colour100.py  the 100x colour of a flake against the colour model (which of the 10x colour's thicknesses it is)
    page.py       the results page (index.html)
"""
