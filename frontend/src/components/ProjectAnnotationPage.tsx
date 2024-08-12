import { ChangeEvent, FC, useCallback, useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';

import { useAddAnnotation, useGetElementById, useGetNextElementId } from '../core/api';
import { useUpdateSimpleModel } from '../core/api';
import { useAuth } from '../core/auth';
import { useAppContext } from '../core/context';
import { useNotifications } from '../core/notifications';
import { ElementOutModel } from '../types';
import { LabelsManagement } from './LabelsManagement';
import { SimpleModelManagement } from './SimpleModelManagement';
import { ProjectPageLayout } from './layout/ProjectPageLayout';

export const ProjectAnnotationPage: FC = () => {
  const { projectName, elementId } = useParams();
  const { authenticatedUser } = useAuth();
  const { notify } = useNotifications();
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

  // be sure to have a scheme selected
  if (!projectName) return null;
  if (!authenticatedUser?.username) return null;
  if (!currentScheme) {
    notify({ type: 'warning', message: 'You need to select first a scheme' });
    navigate(`/projects/${projectName}`);
    return null;
  }

  // hooks to manage element
  const { getNextElementId } = useGetNextElementId(projectName, currentScheme);
  const { getElementById } = useGetElementById(projectName, currentScheme);

  // hooks to manage annotation
  const { addAnnotation } = useAddAnnotation(
    projectName,
    currentScheme,
    authenticatedUser?.username,
  );

  // define parameters for configuration panels
  const availableFeatures = project?.features.available ? project?.features.available : [];
  const availableSimpleModels = project?.simplemodel.options ? project?.simplemodel.options : {};
  const currentModel = project?.simplemodel.available[authenticatedUser?.username][currentScheme]
    ? project?.simplemodel.available[authenticatedUser?.username][currentScheme]
    : { name: 'No simplemodel trained' };
  const availableSamples = project?.next.sample ? project?.next.sample : [];
  const availableLabels = currentScheme && project ? project?.schemes.available[currentScheme] : [];
  // available methods depend if there is a simple model trained for the user/scheme
  // TO TEST, and in the future change the API if possible
  var availableModes = project?.simplemodel.available[authenticatedUser.username][currentScheme]
    ? project?.next.methods
    : project?.next.methods_min;

  // manage the hide/visible menu for the label
  const [selectedMode, setSelectedMode] = useState('');
  const handleSelectChangeMode = (e: ChangeEvent<HTMLSelectElement>) => {
    selectionConfig.mode = e.target.value;
    setSelectedMode(e.target.value);
  };

  // update the selection config when the user change a menu
  useEffect(() => {
    setAppContext((prev) => ({ ...prev, selectionConfig: selectionConfig }));
    console.log('Update selectionConfig');
  }, [selectionConfig]);

  const navigateToNextElement = useCallback(async () => {
    // change url using the new elementId
    // const newElementId = await  apiCall()
    //navigate('/projects/${projectName}/annotate/newid');
    getNextElementId(selectionConfig).then((nextElementId) => {
      if (nextElementId) navigate(`/projects/${projectName}/annotate/${nextElementId}`);
    });
  }, [projectName, navigate]);

  useEffect(() => {
    if (elementId === undefined) {
      getNextElementId(selectionConfig).then((nextElementId) => {
        if (nextElementId) navigate(`/projects/${projectName}/annotate/${nextElementId}`);
      });
    } else {
      //fetch element information (text and labels)
      getElementById(elementId).then(setElement);
    }
  }, [elementId]);

  const handleDisplayPrediction = () => {
    selectionConfig.displayPrediction = !selectionConfig.displayPrediction;
  };

  // hooks to update simplemodel
  const [updatedSimpleModel, setUpdatedSimpleModel] = useState(false);

  // use a memory to only update once
  const { updateSimpleModel } = useUpdateSimpleModel(projectName, currentScheme);
  useEffect(() => {
    if (!updatedSimpleModel && currentModel && history.length % freqRefreshSimpleModel == 0) {
      console.log(currentModel);
      console.log('Update active learning model');
      setUpdatedSimpleModel(true);
      updateSimpleModel(currentModel);
    }
    if (updatedSimpleModel && history.length % freqRefreshSimpleModel != 0)
      setUpdatedSimpleModel(false);
    // TODO UPDATE SIMPLEMODEL
  }),
    [history];

  return (
    <ProjectPageLayout projectName={projectName} currentAction="annotate">
      <div className="container-fluid">
        <div className="row">
          <h2 className="subsection">Annotation</h2>
          <span className="explanations">Configure selection mode and annotate data</span>
        </div>
        <div className="row">
          <div className="col-6 ">
            <details className="custom-details">
              <summary className="custom-summary">Configure selection mode</summary>
              <label>Selection mode</label>
              <select onChange={handleSelectChangeMode}>
                {availableModes ? availableModes : [].map((e, i) => <option key={i}>{e}</option>)}
              </select>
              {selectedMode == 'maxprob' && (
                <div>
                  <label>Label</label>
                  <select onChange={(e) => (selectionConfig.label = e.target.value)}>
                    {availableLabels.map((e, i) => (
                      <option key={i}>{e}</option>
                    ))}{' '}
                  </select>
                </div>
              )}
              <br></br>
              <label>On</label>
              <select onChange={(e) => (selectionConfig.sample = e.target.value)}>
                {availableSamples.map((e, i) => (
                  <option key={i}>{e}</option>
                ))}{' '}
              </select>
              <div>
                Current model :{' '}
                {currentModel ? JSON.stringify(currentModel.name) : 'No model trained'}
              </div>
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
                    onChange={handleDisplayPrediction}
                    style={{ marginRight: '10px' }}
                  />
                  Display prediction
                </label>
              </div>
            </details>
          </div>
        </div>
      </div>

      <div className="row">
        <div className="col-10 annotation-frame my-4">{element?.text}</div>

        {
          //display proba
          selectionConfig.displayPrediction && (
            <div className="d-flex mb-2 justify-content-center display-prediction">
              Predicted label : {element?.predict.label} (proba: {element?.predict.proba})
            </div>
          )
        }
      </div>
      <div className="row">
        <div className="d-flex flex-wrap gap-2 justify-content-center">
          {availableLabels.map((i) => (
            <button
              key={i}
              value={i}
              className="btn btn-primary grow-1"
              onClick={(e) => {
                if (elementId) {
                  addAnnotation(elementId, e.currentTarget.value).then(navigateToNextElement);
                  history.push(elementId);
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
            projectName={projectName}
            currentScheme={currentScheme}
            availableLabels={availableLabels}
            reFetchCurrentProject={reFetchCurrentProject}
          />
        </div>
      </details>
      <details className="custom-details">
        <summary className="custom-summary">Configure prediction model</summary>
        <div className="row">
          <div className="col">
            <SimpleModelManagement
              projectName={projectName}
              currentScheme={currentScheme}
              currentModel={currentModel}
              availableSimpleModels={availableSimpleModels}
              availableFeatures={availableFeatures}
            />
          </div>
        </div>
      </details>
      <div>{element ? JSON.stringify(element) : 'loading...'}</div>
    </ProjectPageLayout>
  );
};
