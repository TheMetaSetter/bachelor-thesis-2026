```
PROCEDURE CalculateBudgetedVUS(Labels, Scores, Buffers, Budget)

    DECLARE BufferIndex : INTEGER
    DECLARE CurrentBuffer : INTEGER
    DECLARE VUS_PR : REAL
    DECLARE VUS_ROC : REAL
    DECLARE PR_At_Buffer : ARRAY[1:Length(Buffers)] OF REAL
    DECLARE ROC_At_Buffer : ARRAY[1:Length(Buffers)] OF REAL
    DECLARE BufferedLabels : ARRAY[1:Length(Labels)] OF REAL
    DECLARE Curve : ARRAY OF RECORD

    FOR BufferIndex ← 1 TO Length(Buffers)

        CurrentBuffer ← Buffers[BufferIndex]

        BufferedLabels ← ApplyVUSBuffer(Labels, CurrentBuffer)

        Curve ← BuildRangeOperatingCurve(BufferedLabels, Scores)

        ROC_At_Buffer[BufferIndex] ←
            CalculateNormalisedPartialROCArea(Curve, Budget)

        PR_At_Buffer[BufferIndex] ←
            CalculateConstrainedPRArea(Curve, Budget)

    NEXT BufferIndex

    VUS_ROC ← TrapezoidArea(Buffers, ROC_At_Buffer) /
              (Maximum(Buffers) - Minimum(Buffers))

    VUS_PR ← TrapezoidArea(Buffers, PR_At_Buffer) /
             (Maximum(Buffers) - Minimum(Buffers))

    OUTPUT VUS_PR
    OUTPUT VUS_ROC

ENDPROCEDURE
```

```
FUNCTION CalculateNormalisedPartialROCArea(Curve, Budget) RETURNS REAL

    DECLARE AllowedCurve : ARRAY OF RECORD
    DECLARE PartialArea : REAL
    DECLARE BoundaryTPR : REAL

    AllowedCurve ← AllPointsWithFPRAtMost(Curve, Budget)

    IF Last(AllowedCurve).FPR < Budget THEN

        BoundaryTPR ← InterpolateTPRAtFPR(Curve, Budget)

        AddPoint(AllowedCurve, Budget, BoundaryTPR)

    ENDIF

    PartialArea ← TrapezoidArea(FPRValues(AllowedCurve),
                                TPRValues(AllowedCurve))

    RETURN PartialArea / Budget

ENDFUNCTION
```

```
FUNCTION CalculateConstrainedPRArea(Curve, Budget) RETURNS REAL

    DECLARE AllowedCurve : ARRAY OF RECORD
    DECLARE Index : INTEGER
    DECLARE PRArea : REAL
    DECLARE RecallChange : REAL

    AllowedCurve ← AllPointsWithFPRAtMost(Curve, Budget)

    PRArea ← 0

    FOR Index ← 2 TO Length(AllowedCurve)

        RecallChange ← AllowedCurve[Index].Recall -
                        AllowedCurve[Index - 1].Recall

        IF RecallChange > 0 THEN

            PRArea ← PRArea +
                     AllowedCurve[Index].Precision * RecallChange

        ENDIF

    NEXT Index

    RETURN PRArea

ENDFUNCTION
```