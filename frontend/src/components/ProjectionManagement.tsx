import { pick } from 'lodash';
import { FC, useCallback, useEffect, useMemo, useState } from 'react';
import { Controller, SubmitHandler, useForm } from 'react-hook-form';
import { useNavigate } from 'react-router-dom';
import Select from 'react-select';

import chroma from 'chroma-js';
import classNames from 'classnames';
import { Modal } from 'react-bootstrap';
import { FaLock, FaPlusCircle } from 'react-icons/fa';
import { FaGear } from 'react-icons/fa6';
import { HiOutlineQuestionMarkCircle } from 'react-icons/hi';
import { Tooltip } from 'react-tooltip';
import { ModelParametersTab } from '../components/ModelParametersTab';
import {
  useAddAnnotation,
  useGetElementById,
  useGetProjectionData,
  useUpdateProjection,
} from '../core/api';
import { useAuth } from '../core/auth';
import { useAppContext } from '../core/context';
import { useNotifications } from '../core/notifications';
import { ElementOutModel, ProjectionParametersModel } from '../types';
import { MulticlassInput } from './Annotation/MulticlassInput';
import { MultilabelInput } from './Annotation/MultilabelInput';
import { ButtonNewFeature } from './ButtonNewFeature';
import { ProjectionVizSigma } from './ProjectionVizSigma';
import { MarqueBoundingBox } from './ProjectionVizSigma/MarqueeController';
import { StopProcessButton } from './StopProcessButton';

interface ProjectionManagementProps {
  projectName: string | null;
  projectSlug?: string;
  currentScheme: string | null;
  availableFeatures: string[];
  currentElementId?: string;
}

// define the component
export const ProjectionManagement: FC<ProjectionManagementProps> = ({
  projectName,
  currentScheme,
  availableFeatures,
  currentElementId,
}) => {
  // hook for all the parameters
  const {
    appContext: {
      currentProject: project,
      currentProjection,
      selectionConfig,
      isComputing,
      labelColorMapping,
      activeModel,
    },
    setAppContext,
  } = useAppContext();
  const navigate = useNavigate();
  const { notify } = useNotifications();

  const { authenticatedUser } = useAuth();
  const { getElementById } = useGetElementById();

  // fetch projection data with the API (null if no model)
  const { projectionData, reFetchProjectionData } = useGetProjectionData(
    projectName,
    currentScheme,
    activeModel || null,
  );

  // states for dynamic interactions
  const [forceRefresh, setForceRefresh] = useState<boolean>(false);
  const [showComputeNewProjection, setShowComputeNewProjection] = useState<boolean>(false);
  const [showParameters, setShowParameters] = useState<boolean>(false);

  // unique labels
  const uniqueLabels = projectionData ? [...new Set(projectionData.labels)] : [];
  const colormap = chroma.scale('Paired').colors(uniqueLabels.length);

  // form management
  const availableProjections = useMemo(() => project?.projections, [project?.projections]);

  const { register, handleSubmit, watch, control, reset } = useForm<ProjectionParametersModel>({
    defaultValues: {
      method: 'umap',
      parameters: {
        //common
        n_components: 2,
        // T-SNE
        perplexity: 30,
        learning_rate: 'auto',
        init: 'random',
        // UMAP
        metric: 'cosine',
        n_neighbors: 15,
        min_dist: 0.1,
      },
      // Normalize
      normalize_features: false,
    },
  });
  const selectedMethod = watch('method'); // state for the model selected to modify parameters

  // available features
  const features = availableFeatures.map((e) => ({ value: e, label: e }));

  // action when form validated
  const { updateProjection } = useUpdateProjection(projectName, currentScheme);
  const onSubmit: SubmitHandler<ProjectionParametersModel> = async (formData) => {
    // fromData has all fields whatever the selected method

    // discard unrelevant fields depending on selected method
    const relevantParams =
      selectedMethod === 'tsne'
        ? ['perplexity', 'n_components', 'learning_rate', 'init']
        : selectedMethod === 'umap'
          ? ['n_neighbors', 'min_dist', 'n_components']
          : [];
    const params = pick(formData.parameters, relevantParams);
    const data = { ...formData, params };
    const watchedFeatures = watch('features');
    if (watchedFeatures.length == 0) {
      notify({ type: 'error', message: 'Please select at least one feature' });
      return;
    }
    await updateProjection(data);
    reset();
    setShowComputeNewProjection(false);
  };

  useEffect(() => {
    if (projectionData) {
      const labeledColors = uniqueLabels.reduce<Record<string, string>>(
        (acc, label, index: number) => {
          acc[label as string] = colormap[index];
          return acc;
        },
        {},
      );
      setAppContext((prev) => ({ ...prev, labelColorMapping: labeledColors }));
    }
  }, [projectionData]);

  // manage projection refresh (could be AMELIORATED)
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

    // case if the projection changed (the available projection in the server is different from the one in the app state)
    if (
      authenticatedUser &&
      currentProjection &&
      currentProjection.status != availableProjections?.available[authenticatedUser?.username]
    ) {
      reFetchProjectionData();
      setAppContext((prev) => ({ ...prev, currentProjection: projectionData || undefined }));
    }

    // After annotating on the fly, force refresh so that the visualisation matches the most recent
    if (authenticatedUser && currentProjection && forceRefresh) {
      reFetchProjectionData();
      setAppContext((prev) => ({ ...prev, currentProjection: projectionData || undefined }));
      setForceRefresh(false);
    }
  }, [
    availableProjections?.available,
    authenticatedUser,
    currentProjection,
    reFetchProjectionData,
    projectionData,
    setAppContext,
    setForceRefresh,
  ]);

  // element to display
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

  type Feature = {
    label: string;
    value: string;
  };
  const filterFeatures = (features: Feature[]) => {
    const filtered = features.filter((e) => /embeddings|fasttext/i.test(e.label));
    return filtered;
  };
  const defaultFeatures = filterFeatures(features);

  // define parameters for configuration panels
  const availableLabels =
    currentScheme && project && project.schemes.available[currentScheme]
      ? project.schemes.available[currentScheme].labels
      : [];
  const [kindScheme] = useState<string>(
    currentScheme && project && project.schemes.available[currentScheme]
      ? project.schemes.available[currentScheme].kind || 'multiclass'
      : 'multiclass',
  );
  // post an annotation
  // hooks to manage annotation
  const { addAnnotation } = useAddAnnotation(projectName || null, currentScheme || null, 'train');

  const postAnnotation = useCallback(
    (label: string | null, elementId?: string) => {
      if (elementId) {
        addAnnotation(elementId, label, '', '');
        setSelectedId(undefined);
        notify({ type: 'success', message: `Annotation added for ${elementId}` });
        setForceRefresh(true);
      }
    },
    [addAnnotation, setSelectedId, notify],
  );

  return (
    <div className="explore-container">
      <div>
        {!isComputing ? (
          <div>
            <button
              onClick={() => setShowComputeNewProjection(true)}
              className="btn-primary-action mb-4"
            >
              <FaPlusCircle size={20} className="me-1" /> Compute new projection
            </button>
          </div>
        ) : (
          <StopProcessButton projectSlug={projectName} />
        )}
        {projectionData && labelColorMapping && (
          <>
            <button className="btn-secondary-action" onClick={() => setShowParameters(true)}>
              <FaGear size={18} />
              Parameters
            </button>
            {(selectionConfig.frame || []).length > 0 && (
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
                <FaLock /> Lock on selection
                <a className="lockhelp">
                  <HiOutlineQuestionMarkCircle />
                </a>
                <Tooltip anchorSelect=".lockhelp" place="top">
                  Once a vizualisation computed, you can use the square tool to select an area (or
                  remove the square).<br></br> Then you can lock the selection, and only elements in
                  the selected area will be available for annoation.
                </Tooltip>
              </label>
            )}
          </>
        )}
      </div>
      {projectionData && labelColorMapping && (
        <div className="explore-viz-container">
          <div className="explore-viz-column">
            <ProjectionVizSigma
              data={projectionData}
              selectedId={currentElementId}
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
              labelColorMapping={labelColorMapping}
            />
          </div>

          <div className={classNames('explore-annotation-column', selectedElement && 'active')}>
            {selectedElement ? (
              <>
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
                <h5 className="subsection">Annotate this element</h5>
                <div className="annotation-block force-one-column-layout">
                  {kindScheme == 'multiclass' && (
                    <MulticlassInput
                      elementId={selectedElement.element_id}
                      element={selectedElement}
                      postAnnotation={postAnnotation}
                      labels={availableLabels}
                      phase="train"
                    />
                  )}
                  {kindScheme == 'multilabel' && (
                    <MultilabelInput
                      elementId={selectedElement.element_id}
                      postAnnotation={postAnnotation}
                      labels={availableLabels}
                    />
                  )}
                </div>
              </>
            ) : (
              <div className="explanations ">Click on an element to display its content</div>
            )}
          </div>
        </div>
      )}

      <Modal
        show={showComputeNewProjection}
        onHide={() => setShowComputeNewProjection(false)}
        size="xl"
        id="viz-projection"
      >
        <Modal.Header closeButton>
          <Modal.Title>Compute a new projection</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <form onSubmit={handleSubmit(onSubmit)}>
            {' '}
            <label htmlFor="features">Select features</label>
            <div style={{ flex: '1 1 auto' }}>
              <Controller
                name="features"
                control={control}
                defaultValue={defaultFeatures.map((e) => e.value)}
                render={({ field: { onChange, value } }) => (
                  <Select
                    options={features}
                    isMulti
                    value={features.filter((option) => value.includes(option.value))}
                    onChange={(selectedOptions) => {
                      onChange(
                        selectedOptions ? selectedOptions.map((option) => option.value) : [],
                      );
                    }}
                  />
                )}
              />
            </div>
            <ButtonNewFeature projectSlug={projectName || ''} />
            <details>
              <summary>Advanced parameters</summary>
              <label htmlFor="model">Select a model</label>
              <select id="model" {...register('method')}>
                {Object.keys(availableProjections?.options || {}).map((e) => (
                  <option key={e} value={e}>
                    {e}
                  </option>
                ))}{' '}
              </select>
              {availableProjections?.options && selectedMethod == 'tsne' && (
                <>
                  <label htmlFor="perplexity">perplexity</label>
                  <input
                    type="number"
                    step="1"
                    id="perplexity"
                    {...register('parameters.perplexity', { valueAsNumber: true })}
                  ></input>
                  <label>Learning rate</label>
                  <select {...register('parameters.learning_rate')}>
                    <option key="auto" value="auto">
                      auto
                    </option>
                  </select>
                  <label>Init</label>
                  <select {...register('parameters.init')}>
                    <option key="random" value="random">
                      random
                    </option>
                  </select>
                </>
              )}
              {availableProjections?.options && selectedMethod == 'umap' && (
                <>
                  <label htmlFor="n_neighbors">n_neighbors</label>
                  <input
                    type="number"
                    step="1"
                    id="n_neighbors"
                    {...register('parameters.n_neighbors', { valueAsNumber: true })}
                  ></input>
                  <label htmlFor="min_dist">min_dist</label>
                  <input
                    type="number"
                    id="min_dist"
                    step="0.01"
                    {...register('parameters.min_dist', { valueAsNumber: true })}
                  ></input>
                </>
              )}
              <input type="checkbox" {...register('normalize_features')} />
              <label>Feature scaling</label>
            </details>
            <button className="btn-submit">Compute</button>
          </form>
        </Modal.Body>
      </Modal>
      <Modal show={showParameters} id="parameters-modal" onHide={() => setShowParameters(false)}>
        <Modal.Header closeButton>
          <Modal.Title>Parameters of the current visualisation</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <ModelParametersTab
            params={
              projectionData?.parameters
                ? {
                    method: projectionData.parameters.method,
                    features: projectionData.parameters.features,
                    ...projectionData.parameters.parameters,
                  }
                : ({} as Record<string, unknown>)
            }
          />
        </Modal.Body>
      </Modal>
    </div>
  );
};
