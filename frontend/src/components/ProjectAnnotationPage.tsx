import { FC, useCallback, useEffect, useMemo, useState } from 'react';
import Tab from 'react-bootstrap/Tab';
import Tabs from 'react-bootstrap/Tabs';
import { IoMdSkipBackward } from 'react-icons/io';
import { Link, useNavigate, useParams } from 'react-router-dom';
import {
  useAddAnnotation,
  useGetElementById,
  useGetNextElementId,
  useStatistics,
  useUpdateSimpleModel,
} from '../core/api';
import { useAuth } from '../core/auth';
import { useAppContext } from '../core/context';
import { ElementOutModel } from '../types';
import { LabelsManagement } from './LabelsManagement';
import { ProjectionManagement } from './ProjectionManagement';
import { SelectCurrentScheme } from './SchemesManagement';
import { SelectionManagement } from './SelectionManagement';
import { SimpleModelManagement } from './SimpleModelManagement';
import { ProjectPageLayout } from './layout/ProjectPageLayout';

/**
 * Annotation page
 */
export const ProjectAnnotationPage: FC = () => {
  // parameters
  const { projectName, elementId } = useParams();
  const { authenticatedUser } = useAuth();
  const {
    appContext: {
      currentScheme,
      reFetchCurrentProject,
      currentProject: project,
      selectionConfig,
      displayConfig,
      freqRefreshSimpleModel,
      history,
      phase,
    },
    setAppContext,
  } = useAppContext();

  const navigate = useNavigate();
  const [element, setElement] = useState<ElementOutModel | null>(null); //state for the current element

  // hooks to manage element
  const { getNextElementId } = useGetNextElementId(
    projectName || null,
    currentScheme || null,
    selectionConfig,
    history,
    phase,
  );
  const { getElementById } = useGetElementById(projectName || null, currentScheme || null);

  // hooks to manage annotation
  const { addAnnotation } = useAddAnnotation(projectName || null, currentScheme || null, phase);

  // define parameters for configuration panels
  const availableFeatures = project?.features.available ? project?.features.available : [];
  const availableSimpleModels = project?.simplemodel.options ? project?.simplemodel.options : {};
  const currentModel = useMemo(() => {
    return authenticatedUser &&
      currentScheme &&
      project?.simplemodel.available[authenticatedUser?.username]?.[currentScheme]
      ? project?.simplemodel.available[authenticatedUser?.username][currentScheme]
      : null;
  }, [project, currentScheme, authenticatedUser]);
  const availableLabels = useMemo(() => {
    return currentScheme && project ? project.schemes.available[currentScheme] || [] : [];
  }, [project, currentScheme]);
  // available methods depend if there is a simple model trained for the user/scheme
  // TO TEST, and in the future change the API if possible

  // get statistics to display (TODO : try a way to avoid another request ?)
  const { statistics, reFetchStatistics } = useStatistics(
    projectName || null,
    currentScheme || null,
  );

  // react to URL param change
  useEffect(() => {
    if (elementId === undefined) {
      // add fetch current selectionConfig in the hook code
      getNextElementId().then((nextElementId) => {
        if (nextElementId) navigate(`/projects/${projectName}/annotate/${nextElementId}`);
        else {
          navigate(`/projects/${projectName}/annotate/noelement`);
        }
      });
    } else {
      //fetch element information (text and labels)
      getElementById(elementId, phase).then((element) => {
        if (element) setElement(element);
        else setTimeout(() => {}, 1000);
      });
      reFetchStatistics();
    }
  }, [
    elementId,
    getNextElementId,
    getElementById,
    navigate,
    phase,
    projectName,
    reFetchStatistics,
  ]);

  // hook to fetch a next element when selectionConfig changes
  useEffect(() => {
    getNextElementId().then((nextElementId) => {
      if (nextElementId) navigate(`/projects/${projectName}/annotate/${nextElementId}`);
      else navigate(`/projects/${projectName}/annotate/noelement`);
    });
  }, [selectionConfig, elementId, getNextElementId, projectName, navigate]);

  // hooks to update simplemodel
  const [updatedSimpleModel, setUpdatedSimpleModel] = useState(false);

  // use a memory to only update once
  const { updateSimpleModel } = useUpdateSimpleModel(projectName || null, currentScheme || null);

  useEffect(() => {
    if (!updatedSimpleModel && currentModel && history.length % freqRefreshSimpleModel == 0) {
      setUpdatedSimpleModel(true);
      updateSimpleModel(currentModel);
    }
    if (updatedSimpleModel && history.length % freqRefreshSimpleModel != 0)
      setUpdatedSimpleModel(false);
  }, [
    history,
    updateSimpleModel,
    setUpdatedSimpleModel,
    currentModel,
    freqRefreshSimpleModel,
    updatedSimpleModel,
  ]);

  // generic method to apply a chosen label to an element
  const applyLabel = useCallback(
    (label: string, elementId?: string) => {
      if (elementId) {
        setAppContext((prev) => ({ ...prev, history: [...prev.history, elementId] }));
        addAnnotation(elementId, label).then(() =>
          // redirect to next element by redirecting wihout any id
          // thus the getNextElementId query will be dont after the appcontext is reloaded
          navigate(`/projects/${projectName}/annotate/`),
        );
        // does not do nothing as we remount through navigate reFetchStatistics();
      }
    },
    [setAppContext, addAnnotation, navigate, projectName],
  );

  const handleKeyboardEvents = useCallback(
    (ev: KeyboardEvent) => {
      availableLabels.forEach((label, i) => {
        if (ev.code === `Digit` + (i + 1) || ev.code === `Numpad` + (i + 1)) {
          applyLabel(label, elementId);
        }
      });
    },
    [availableLabels, applyLabel, elementId],
  );

  useEffect(() => {
    // manage keyboard shortcut if less than 10 label
    if (availableLabels.length > 0 && availableLabels.length < 10) {
      document.addEventListener('keydown', handleKeyboardEvents);
    }

    return () => {
      if (availableLabels.length > 0 && availableLabels.length < 10) {
        document.removeEventListener('keydown', handleKeyboardEvents);
      }
    };
  }, [availableLabels, handleKeyboardEvents]);

  return (
    <ProjectPageLayout projectName={projectName || null} currentAction="annotate">
      <div className="container-fluid">
        <div className="row mb-3 mt-3">
          {phase == 'test' && (
            <div className="alert alert-warning">
              Test mode activated - you are annotating test set
            </div>
          )}
          {phase != 'test' && (
            <Tabs id="panel2" className="mb-3" defaultActiveKey="scheme">
              <Tab eventKey="scheme" title="Current scheme">
                <div className="row">
                  <div className="col-6">
                    <SelectCurrentScheme />
                  </div>
                  <div className="col-6">
                    {statistics ? (
                      <span className="badge text-bg-light  mt-2">
                        Count :{' '}
                        {`${statistics[phase == 'test' ? 'test_annotated_n' : 'train_annotated_n']} / ${statistics[phase == 'test' ? 'test_set_n' : 'train_set_n']}`}
                      </span>
                    ) : (
                      ''
                    )}
                  </div>{' '}
                </div>
              </Tab>
              <Tab eventKey="selection" title="Selection mode">
                <SelectionManagement />
              </Tab>
              <Tab eventKey="parameters" title="Display parameters">
                <label style={{ display: 'block', marginBottom: '10px' }}>
                  <input
                    type="checkbox"
                    checked={displayConfig.displayPrediction || false}
                    onChange={(_) => {
                      setAppContext((prev) => ({
                        ...prev,
                        displayConfig: {
                          ...displayConfig,
                          displayPrediction: displayConfig.displayPrediction
                            ? !displayConfig.displayPrediction
                            : true,
                        },
                      }));
                    }}
                    style={{ marginRight: '10px' }}
                  />
                  Display prediction
                </label>
                <label style={{ display: 'block', marginBottom: '10px' }}>
                  <input
                    type="checkbox"
                    checked={displayConfig.displayContext}
                    onChange={(_) => {
                      setAppContext((prev) => ({
                        ...prev,
                        displayConfig: {
                          ...displayConfig,
                          displayContext: !displayConfig.displayContext,
                        },
                      }));
                    }}
                    style={{ marginRight: '10px' }}
                  />
                  Display informations
                </label>
                <label style={{ display: 'block', marginBottom: '10px' }}>
                  Text frame size
                  <span>Min: 25%</span>
                  <input
                    type="range"
                    min="25"
                    max="100"
                    className="form-input"
                    onChange={(e) => {
                      setAppContext((prev) => ({
                        ...prev,
                        displayConfig: {
                          ...displayConfig,
                          frameSize: Number(e.target.value),
                        },
                      }));
                      console.log(displayConfig.frameSize);
                    }}
                    style={{ marginRight: '10px' }}
                  />
                  <span>Max: 100%</span>
                </label>
              </Tab>
            </Tabs>
          )}
        </div>
      </div>

      {
        // back button
      }

      {
        // display content
      }
      <div className="row">
        {element?.text && (
          <div
            className="col-11 annotation-frame my-4"
            style={{ height: `${displayConfig.frameSize}vh` }}
          >
            <span>{element?.text.slice(0, element?.limit as number)}</span>
            <span className="text-out-context">
              {element?.text.slice(element?.limit as number)}
            </span>
          </div>
        )}

        {
          //display proba
          (displayConfig.displayPrediction || false) && (
            <div className="d-flex mb-2 justify-content-center display-prediction">
              Predicted label : {element?.predict.label} (proba: {element?.predict.proba})
            </div>
          )
        }
        {
          //display informations
          displayConfig.displayContext && (
            <div className="d-flex mb-2 justify-content-center display-prediction">
              Context : {JSON.stringify(element?.context)}
            </div>
          )
        }
      </div>
      <div className="row">
        <div className="d-flex flex-wrap gap-2 justify-content-center">
          <Link
            to={`/projects/${projectName}/annotate/${history[history.length - 1]}`}
            className="btn btn-outline-secondary"
            onClick={() => {
              setAppContext((prev) => ({ ...prev, history: prev.history.slice(0, -1) }));
            }}
          >
            <IoMdSkipBackward />
          </Link>

          {
            // display buttons for label
            availableLabels.map((i, e) => (
              <button
                type="button"
                key={i}
                value={i}
                className="btn btn-primary grow-1"
                onClick={(e) => {
                  applyLabel(e.currentTarget.value, elementId);
                }}
              >
                {i} <span className="badge text-bg-secondary">{e + 1}</span>
              </button>
            ))
          }
        </div>
      </div>
      <div className="mt-5">
        {phase != 'test' && (
          <Tabs id="panel2" className="mb-3" defaultActiveKey="description">
            <Tab eventKey="description" title="Annotations">
              <span className="explanations">
                Configure the selection mode, train prediction model to enable active learning
              </span>
            </Tab>
            <Tab eventKey="labels" title="Labels">
              <LabelsManagement
                projectName={projectName || null}
                currentScheme={currentScheme || null}
                availableLabels={availableLabels}
                reFetchCurrentProject={reFetchCurrentProject || (() => null)}
              />
            </Tab>
            <Tab eventKey="prediction" title="Prediction">
              <SimpleModelManagement
                projectName={projectName || null}
                currentScheme={currentScheme || null}
                availableSimpleModels={availableSimpleModels}
                availableFeatures={availableFeatures}
              />
            </Tab>
            <Tab eventKey="projection" title="Projection">
              <ProjectionManagement />
            </Tab>
          </Tabs>
        )}
      </div>
    </ProjectPageLayout>
  );
};
