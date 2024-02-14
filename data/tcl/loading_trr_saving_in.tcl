    # Load the gro and trr and with optinal number of frames
    set mol [mol new @GRO  type @TYPE waitfor all]
    puts "Attempting to load file: $mol"
    set distance 5

    mol addfile @TRR type trr first @FIRST last @LAST step @STEP\
    waitfor all molid $mol

    set nf [molinfo top get numframes]
    puts " The numbers of frames are $nf"
    for {set i 0} {$i < $nf} {incr i} {
        animate goto $i
        set inRadiusResidues [atomselect top "@ATOMSELCT"]
        set resIDs [$inRadiusResidues get residue]
        if {[llength $resIDs] > 0} {
            set uniqueResIDs [lsort -unique $resIDs]
            set completeInRadiusResidues\
            [atomselect top "@FINALSELECT $uniqueResIDs"]
            $completeInRadiusResidues @WRITE "@OUTNAME_${i}.@TYPE"
            $completeInRadiusResidues delete
        } else {
            puts "No residues within $distance Å of APT or COR in frame $i"
        }
        $inRadiusResidues delete
    }

    exit
