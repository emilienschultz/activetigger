import { FC, useCallback, useEffect, useMemo, useState } from 'react';

import { FaLock } from 'react-icons/fa';
import { HiOutlineQuestionMarkCircle } from 'react-icons/hi';
import { useNavigate } from 'react-router-dom';
import { Tooltip } from 'react-tooltip';
import { useGetElementById, useGetProjectionData } from '../../core/api';
import { useAuth } from '../../core/auth';
import { useAppContext } from '../../core/context';
import { ElementOutModel } from '../../types';
import { ProjectionVizSigma } from '../ProjectionVizSigma';
import { MarqueBoundingBox } from '../ProjectionVizSigma/MarqueeController';

interface DisplayProjectionProps {
  projectName: string | null;
  currentScheme: string | null;
  currentElement?: ElementOutModel | null;
}

// define the component
export const DisplayProjection: FC<DisplayProjectionProps> = ({
  projectName,
  currentScheme,
  currentElement,
}) => {
  // hook for all the parameters
  const {
    appContext: {
      currentProject: project,
      currentProjection,
      selectionConfig,
      labelColorMapping,
      activeModel,
    },
    setAppContext,
  } = useAppContext();
  const { authenticatedUser } = useAuth();
  const navigate = useNavigate();

  // fetch projection data with the API (null if no model)
  const { projectionData, reFetchProjectionData } = useGetProjectionData(
    projectName,
    currentScheme,
    activeModel || null,
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

  // element to display
  const { getElementById } = useGetElementById();
  const [selectedElement, setSelectedElement] = useState<ElementOutModel | null>(null);
  const setSelectedId = useCallback(
    (id?: string) => {
      if (id)
        getElementById(id, 'train').then((element) => {
          setSelectedElement(element || null);
        });
      else setSelectedElement(null);
    },
    [getElementById, setSelectedElement],
  );

  // if the element changes from outside, update the selectedElement
  useEffect(() => {
    setSelectedElement(currentElement || null);
  }, [currentElement]);

  return (
    <div style={{ width: '80%' }}>
      {currentProjection ? (
        <>
          <div className="my-2">
            <label style={{ display: 'block' }}>
              <input
                type="checkbox"
                checked={selectionConfig.frameSelection}
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

          <div className="d-flex flex-column">
            <div>
              <ProjectionVizSigma
                data={currentProjection}
                selectedId={selectedElement?.element_id || undefined}
                setSelectedId={setSelectedId}
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
            <>
              {selectedElement ? (
                <div
                  style={{
                    overflowY: 'auto',
                    maxHeight: '80vh',
                  }}
                  className="mx-4"
                >
                  <a
                    className="badge m-0 p-1"
                    onClick={() =>
                      navigate(`/projects/${projectName}/tag/${selectedElement.element_id}?tab=tag`)
                    }
                    style={{ cursor: 'pointer' }}
                  >
                    Text {selectedElement.element_id}
                  </a>
                  <div>{selectedElement.text}</div>
                  <details>
                    <summary>Previous annotations:</summary>
                    <ul>
                      {selectedElement.history?.map((e) => {
                        return (
                          <li key={`${e.time}-${e.user}`}>
                            label: {e.label ? e.label : 'label removed'} ({e.time} by {e.user})
                            <br />
                          </li>
                        );
                      })}
                    </ul>
                  </details>
                </div>
              ) : (
                <div className="explanations horizontal center" style={{ flex: '1 1 auto' }}>
                  Click on an element to display its content
                </div>
              )}
            </>
          </div>
        </>
      ) : (
        <>No projection computed</>
      )}
    </div>
  );
};
