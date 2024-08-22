import { ChangeEvent, FC, useCallback, useEffect, useMemo, useState } from 'react';
import { IoMdReturnLeft } from 'react-icons/io';
import { Link, useNavigate, useParams } from 'react-router-dom';

import {
  useAddAnnotation,
  useGetElementById,
  useGetNextElementId,
  useUpdateSimpleModel,
} from '../core/api';
import { useAuth } from '../core/auth';
import { useAppContext } from '../core/context';
import { ElementOutModel } from '../types';
import { LabelsManagement } from './LabelsManagement';
import { ProjectionManagement } from './ProjectionManagement';
import { SelectCurrentScheme } from './SchemesManagement';
import { SimpleModelManagement } from './SimpleModelManagement';
import { ProjectPageLayout } from './layout/ProjectPageLayout';

export const ProjectAnnotationPage: FC = () => {
  const { projectName, elementId } = useParams();
  const { authenticatedUser } = useAuth();
  const {
    appContext: {
      currentScheme,
      reFetchCurrentProject,
      currentProject: project,
      selectionConfig,
      freqRefreshSimpleModel,
      history,
    },
    setAppContext,
  } = useAppContext();

  const navigate = useNavigate();
  const [element, setElement] = useState<ElementOutModel | null>(null); //state for the current element

  // hooks to manage element
  const { getNextElementId } = useGetNextElementId(projectName || null, currentScheme || null);
  const { getElementById } = useGetElementById(projectName || null, currentScheme || null);

  // hooks to manage annotation
  const { addAnnotation } = useAddAnnotation(projectName || null, currentScheme || null);

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
  const availableSamples = project?.next.sample ? project?.next.sample : [];
  const availableLabels = useMemo(() => {
    return currentScheme && project ? project.schemes.available[currentScheme] || [] : [];
  }, [project, currentScheme]);
  // available methods depend if there is a simple model trained for the user/scheme
  // TO TEST, and in the future change the API if possible
  const availableModes =
    authenticatedUser &&
    currentScheme &&
    project?.simplemodel.available[authenticatedUser.username]?.[currentScheme]
      ? project.next.methods
      : project?.next.methods_min
        ? project?.next.methods_min
        : [];

  const elementOutModel = useMemo(() => {
    return {
      element_id: '',
      text: '',
      context: {},
      selection: '',
      info: null,
      predict: {},
      frame: [],
      limit: null,
      history: [],
    };
  }, []);

  const navigateToNextElement = useCallback(async () => {
    getNextElementId(selectionConfig).then((nextElementId) => {
      if (nextElementId) navigate(`/projects/${projectName}/annotate/${nextElementId}`);
      else setElement(elementOutModel);
    });
  }, [projectName, navigate, selectionConfig, getNextElementId, elementOutModel]);

  useEffect(() => {
    if (elementId === undefined) {
      getNextElementId(selectionConfig).then((nextElementId) => {
        if (nextElementId) navigate(`/projects/${projectName}/annotate/${nextElementId}`);
      });
    } else {
      //fetch element information (text and labels)
      getElementById(elementId).then(setElement);
    }
  }, [
    elementId,
    elementOutModel,
    getNextElementId,
    getElementById,
    navigate,
    selectionConfig,
    projectName,
  ]);

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
    // TODO UPDATE SIMPLEMODEL
  }, [
    history,
    updateSimpleModel,
    setUpdatedSimpleModel,
    currentModel,
    freqRefreshSimpleModel,
    updatedSimpleModel,
  ]);

  const handleKeyboardEvents = useCallback(
    (ev: KeyboardEvent) => {
      availableLabels.forEach((label, i) => {
        if (ev.code === `Digit` + (i + 1) || ev.code === `Numpad` + (i + 1)) {
          if (elementId) {
            console.log(label);
            addAnnotation(elementId, label).then(() => navigateToNextElement);
            setAppContext((prev) => ({ ...prev, history: [...history, elementId] }));
          }
        }
      });
    },
    [availableLabels, addAnnotation, setAppContext, elementId, history, navigateToNextElement],
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
        <div className="row">
          <h2 className="subsection">Annotation</h2>
          <span className="explanations">Configure selection mode and annotate data</span>
        </div>
        <div className="row">
          <div className="col-6">
            <SelectCurrentScheme />
          </div>
        </div>
        <div className="row">
          <div className="col-6 ">
            <details className="custom-details">
              <summary className="custom-summary">Configure selection mode</summary>
              <div className="d-flex align-items-center justify-content-between">
                <label>Selection mode</label>
                <select
                  className="form-select w-50"
                  onChange={(e: ChangeEvent<HTMLSelectElement>) => {
                    setAppContext((prev) => ({
                      ...prev,
                      selectionConfig: { ...selectionConfig, mode: e.target.value },
                    }));
                  }}
                >
                  {availableModes.map((e, i) => (
                    <option key={i}>{e}</option>
                  ))}
                </select>
              </div>
              {selectionConfig.mode == 'maxprob' && (
                <div>
                  <label>Label</label>
                  <select
                    onChange={(e) => {
                      setAppContext((prev) => ({
                        ...prev,
                        selectionConfig: { ...selectionConfig, label: e.target.value },
                      }));
                    }}
                  >
                    {availableLabels.map((e, i) => (
                      <option key={i}>{e}</option>
                    ))}{' '}
                  </select>
                </div>
              )}
              <div className="d-flex align-items-center justify-content-between">
                <label>On</label>
                <select
                  className="form-select w-50"
                  onChange={(e) => {
                    setAppContext((prev) => ({
                      ...prev,
                      selectionConfig: { ...selectionConfig, sample: e.target.value },
                    }));
                  }}
                >
                  {availableSamples.map((e, i) => (
                    <option key={i}>{e}</option>
                  ))}{' '}
                </select>
              </div>
              <div className="d-flex align-items-center justify-content-between">
                <label htmlFor="select_regex">Filter</label>
                <input
                  className="form-control w-75"
                  type="text"
                  id="select_regex"
                  placeholder="Enter a regex / CONTEXT= for context"
                  onChange={(e) => {
                    setAppContext((prev) => ({
                      ...prev,
                      selectionConfig: { ...selectionConfig, filter: e.target.value },
                    }));
                  }}
                />
              </div>
              <div>Current model : {currentModel ? currentModel['model'] : 'No model trained'}</div>
            </details>
          </div>
          <div className="col-6 ">
            {' '}
            <details className="custom-details">
              <summary className="custom-summary">Display parameters</summary>
              <div>
                <label style={{ display: 'block', marginBottom: '10px' }}>
                  <input
                    type="checkbox"
                    checked={selectionConfig.displayPrediction}
                    onChange={(_) => {
                      setAppContext((prev) => ({
                        ...prev,
                        selectionConfig: {
                          ...selectionConfig,
                          displayPrediction: !selectionConfig.displayPrediction,
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
                    checked={selectionConfig.displayContext}
                    onChange={(_) => {
                      setAppContext((prev) => ({
                        ...prev,
                        selectionConfig: {
                          ...selectionConfig,
                          displayContext: !selectionConfig.displayContext,
                        },
                      }));
                    }}
                    style={{ marginRight: '10px' }}
                  />
                  Display informations
                </label>
              </div>
            </details>
          </div>
        </div>
      </div>

      {
        // back button
      }

      {
        // display content
      }
      <div className="row">
        <div className="col-10 annotation-frame my-4">
          <span>{element?.text.slice(0, element?.limit as number)}</span>
          <span className="text-out-context">{element?.text.slice(element?.limit as number)}</span>
        </div>

        {
          //display proba
          selectionConfig.displayPrediction && (
            <div className="d-flex mb-2 justify-content-center display-prediction">
              Predicted label : {element?.predict.label} (proba: {element?.predict.proba})
            </div>
          )
        }
        {
          //display informations
          selectionConfig.displayContext && (
            <div className="d-flex mb-2 justify-content-center display-prediction">
              Context : {JSON.stringify(element?.context)}
            </div>
          )
        }
      </div>
      <div className="row">
        <div className="d-flex flex-wrap gap-2 justify-content-center">
          <Link
            to={'/projects/test3/annotate/' + history[history.length - 1]}
            className="btn btn-outline-secondary"
            onClick={() => {
              setAppContext((prev) => ({ ...prev, history: history.slice(0, -1) }));
            }}
          >
            <IoMdReturnLeft />
          </Link>
          {availableLabels.map((i) => (
            <button
              key={i}
              value={i}
              className="btn btn-primary grow-1"
              onClick={(e) => {
                if (elementId) {
                  addAnnotation(elementId, e.currentTarget.value).then(navigateToNextElement);
                  setAppContext((prev) => ({ ...prev, history: [...history, elementId] }));
                  // TODO manage erreur
                }
              }}
            >
              {i}
            </button>
          ))}
        </div>
      </div>
      <hr />
      <details className="custom-details">
        <summary className="custom-summary">Delete, create or replace labels</summary>
        <div className="d-flex align-items-center">
          <LabelsManagement
            projectName={projectName || null}
            currentScheme={currentScheme || null}
            availableLabels={availableLabels}
            reFetchCurrentProject={reFetchCurrentProject || (() => null)}
          />
        </div>
      </details>
      <details className="custom-details">
        <summary className="custom-summary">Configure prediction model</summary>
        <div className="row">
          <div className="col">
            <SimpleModelManagement
              projectName={projectName || null}
              currentScheme={currentScheme || null}
              availableSimpleModels={availableSimpleModels}
              availableFeatures={availableFeatures}
            />
          </div>
        </div>
      </details>
      {project && (
        <details className="custom-details">
          <summary className="custom-summary">Compute projection</summary>
          <div className="row">
            <div className="col">
              <ProjectionManagement />
            </div>
          </div>
        </details>
      )}
    </ProjectPageLayout>
  );
};
