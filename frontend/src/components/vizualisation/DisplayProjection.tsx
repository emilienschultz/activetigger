import { FC, useEffect, useMemo } from 'react';

import { FaLock } from 'react-icons/fa';
import { HiOutlineQuestionMarkCircle } from 'react-icons/hi';
import { Tooltip } from 'react-tooltip';
import { useGetProjectionData } from '../../core/api';
import { useAuth } from '../../core/auth';
import { useAppContext } from '../../core/context';
import { ProjectionVizSigma } from '../ProjectionVizSigma';
import { MarqueBoundingBox } from '../ProjectionVizSigma/MarqueeController';

interface DisplayProjectionProps {
  projectName: string | null;
  currentScheme: string | null;
  elementId?: string;
}

// define the component
export const DisplayProjection: FC<DisplayProjectionProps> = ({
  projectName,
  currentScheme,
  elementId,
}) => {
  // hook for all the parameters
  const {
    appContext: { currentProject: project, currentProjection, selectionConfig, labelColorMapping },
    setAppContext,
  } = useAppContext();
  const { authenticatedUser } = useAuth();

  // fetch projection data with the API (null if no model)
  const { projectionData, reFetchProjectionData } = useGetProjectionData(
    projectName,
    currentScheme,
  );

  // available projections
  const availableProjections = useMemo(() => project?.projections, [project?.projections]);

  // fetch projection if needed and set it in the context
  useEffect(() => {
    // case a first projection is added
    if (
      authenticatedUser &&
      !currentProjection &&
      availableProjections?.available[authenticatedUser?.username]
    ) {
      reFetchProjectionData();
      setAppContext((prev) => ({ ...prev, currentProjection: projectionData || undefined }));
    }
  }, [
    availableProjections?.available,
    authenticatedUser,
    currentProjection,
    reFetchProjectionData,
    projectionData,
    setAppContext,
  ]);

  return (
    <div>
      {currentProjection ? (
        <div className="row align-items-start" style={{ height: '400px', marginBottom: '50px' }}>
          <div className="my-2">
            <label style={{ display: 'block' }}>
              <input
                type="checkbox"
                checked={selectionConfig.frameSelection}
                className="mx-2"
                onChange={(_) => {
                  setAppContext((prev) => ({
                    ...prev,
                    selectionConfig: {
                      ...selectionConfig,
                      frameSelection: !selectionConfig.frameSelection,
                    },
                  }));
                }}
              />
              <span className="lock">
                <FaLock /> Lock on selection
              </span>
              <a className="lockhelp">
                <HiOutlineQuestionMarkCircle />
              </a>
              <Tooltip anchorSelect=".lockhelp" place="top">
                Once a vizualisation computed, you can use the square tool to select an area (or
                remove the square).<br></br> Then you can lock the selection, and only elements in
                the selected area will be available for annoation.
              </Tooltip>
            </label>
          </div>
          <ProjectionVizSigma
            className={`col-12 border h-100`}
            data={currentProjection}
            selectedId={elementId || undefined}
            setSelectedId={(id?: string | undefined) => id}
            frame={selectionConfig.frame}
            setFrameBbox={(bbox?: MarqueBoundingBox) => {
              setAppContext((prev) => ({
                ...prev,
                selectionConfig: {
                  ...selectionConfig,
                  frame: bbox ? [bbox.x.min, bbox.x.max, bbox.y.min, bbox.y.max] : undefined,
                },
              }));
            }}
            labelColorMapping={labelColorMapping || {}}
          />
        </div>
      ) : (
        <>No projection computed</>
      )}
    </div>
  );
};
