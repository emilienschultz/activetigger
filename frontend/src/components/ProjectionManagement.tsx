import { pick } from 'lodash';
import { FC, useEffect, useState } from 'react';
import { Controller, SubmitHandler, useForm } from 'react-hook-form';
import { FaLock } from 'react-icons/fa';
import { useNavigate } from 'react-router-dom';

import Select from 'react-select';
import {
  DomainTuple,
  VictoryAxis,
  VictoryChart,
  VictoryLegend,
  VictoryScatter,
  VictoryTheme,
  VictoryTooltip,
  VictoryZoomContainer,
} from 'victory';

import { LuZoomIn } from 'react-icons/lu';
import { useGetElementById, useGetProjectionData, useUpdateProjection } from '../core/api';
import { useAuth } from '../core/auth';
import { useAppContext } from '../core/context';
import { useNotifications } from '../core/notifications';
import { ElementOutModel, ProjectionInStrictModel, ProjectionModelParams } from '../types';

interface ZoomDomain {
  x?: DomainTuple;
  y?: DomainTuple;
}

const colormap = [
  '#1f77b4', // tab:blue
  '#ff7f0e', // tab:orange
  '#2ca02c', // tab:green
  '#d62728', // tab:red
  '#9467bd', // tab:purple
  '#8c564b', // tab:brown
  '#e377c2', // tab:pink
  '#7f7f7f', // tab:gray
  '#bcbd22', // tab:olive
  '#17becf', // tab:cyan
];

// define the component
export const ProjectionManagement: FC<{ currentElementId: string | null }> = ({
  currentElementId,
}) => {
  // hook for all the parameters
  const {
    appContext: { currentProject: project, currentScheme, currentProjection, selectionConfig },
    setAppContext,
  } = useAppContext();
  const navigate = useNavigate();
  const { notify } = useNotifications();

  const { authenticatedUser } = useAuth();
  const { getElementById } = useGetElementById(
    project?.params.project_slug || null,
    currentScheme || null,
  );

  const projectName = project?.params.project_slug ? project?.params.project_slug : null;

  // fetch projection data with the API (null if no model)
  const { projectionData, reFetchProjectionData } = useGetProjectionData(
    projectName,
    currentScheme,
  );

  // form management
  const availableFeatures = project?.features.available ? project?.features.available : [];
  const availableProjections = project?.projections.options ? project?.projections.options : null;

  const { register, handleSubmit, watch, control } = useForm<ProjectionInStrictModel>({
    defaultValues: {
      method: 'umap',
      features: [],
      params: {
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
    },
  });
  const selectedMethod = watch('method'); // state for the model selected to modify parameters

  // available features
  const features = availableFeatures.map((e) => ({ value: e, label: e }));

  // action when form validated
  const { updateProjection } = useUpdateProjection(projectName, currentScheme);
  const onSubmit: SubmitHandler<ProjectionInStrictModel> = async (formData) => {
    // fromData has all fields whatever the selected method

    // discard unrelevant fields depending on selected method
    const relevantParams =
      selectedMethod === 'tsne'
        ? ['perplexity', 'n_components', 'learning_rate', 'init']
        : selectedMethod === 'umap'
          ? ['n_neighbors', 'min_dist', 'metric', 'n_components']
          : [];
    const params = pick(formData.params, relevantParams) as ProjectionModelParams;
    const data = { ...formData, params };
    const watchedFeatures = watch('features');
    console.log(watchedFeatures);
    if (watchedFeatures.length == 0) {
      notify({ type: 'error', message: 'Please select at least one feature' });
      return;
    }
    await updateProjection(data);
  };

  // scatterplot management for colors
  const [labelColorMapping, setLabelColorMapping] = useState<{ [key: string]: string } | null>(
    null,
  );

  useEffect(() => {
    if (projectionData && !labelColorMapping) {
      const uniqueLabels = projectionData ? [...new Set(projectionData.labels)] : [];
      const labeledColors = uniqueLabels.reduce<Record<string, string>>(
        (acc, label, index: number) => {
          acc[label as string] = colormap[index];
          return acc;
        },
        {},
      );
      setLabelColorMapping(labeledColors);
    }
  }, [projectionData, labelColorMapping]);

  // manage projection refresh (could be AMELIORATED)
  console.log(projectionData);
  useEffect(() => {
    console.log('USEEFFECT');
    // case a first projection is added
    if (
      project &&
      authenticatedUser &&
      !currentProjection &&
      project?.projections.available[authenticatedUser?.username]
    ) {
      reFetchProjectionData();
      setAppContext((prev) => ({ ...prev, currentProjection: projectionData?.status }));
      console.log('Fetch projection data');
    }
    // case if the projection changed
    if (
      authenticatedUser &&
      currentProjection &&
      currentProjection != project?.projections.available[authenticatedUser?.username]
    ) {
      console.log('Refetch projection data');
      reFetchProjectionData();
      setAppContext((prev) => ({ ...prev, currentProjection: projectionData?.status }));
    }
  }, [
    project,
    authenticatedUser,
    currentProjection,
    reFetchProjectionData,
    projectionData,
    setAppContext,
  ]);

  // zoom management
  const initialZoomDomain = {
    x: [-1.5, 1.5] as DomainTuple,
    y: [-1.5, 1.5] as DomainTuple,
  };
  const step = 0.2;

  const [zoomDomain, setZoomDomain] = useState<{ x?: DomainTuple; y?: DomainTuple } | null>(
    initialZoomDomain,
  );

  const handleZoom = (domain: ZoomDomain) => {
    if (!zoomDomain) setZoomDomain(initialZoomDomain);
    setZoomDomain(domain);

    if (domain.x && domain.y) {
      setAppContext((prev) => ({
        ...prev,
        selectionConfig: {
          ...selectionConfig,
          frame: ([] as number[]).concat(
            Object.values(domain.x || []),
            Object.values(domain.y || []),
          ),
        },
      }));
    }
  };
  const handleZoomIn = () => {
    if (zoomDomain && zoomDomain.x && zoomDomain.y) {
      setZoomDomain({
        x: [Number(zoomDomain.x[0]) + step, Number(zoomDomain.x[1]) - step],
        y: [Number(zoomDomain.y[0]) + step, Number(zoomDomain.y[1]) - step],
      });
    }
  };
  const resetZoom = () => {
    setZoomDomain(initialZoomDomain);
  };

  // element to display
  const [selectedElement, setSelectedElement] = useState<ElementOutModel | null>(null);

  //  console.log(project);

  return (
    <div>
      {projectionData && labelColorMapping && (
        <div className="row align-items-start">
          <div className="col-8">
            <div className="d-flex align-items-center justify-content-center">
              <label className="d-flex align-items-center mx-4" style={{ display: 'block' }}>
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
                    // console.log(selectionConfig.frameSelection);
                  }}
                />
                <FaLock />
                {/* Use visualisation frame to lock the selection */}
              </label>
              <button onClick={handleZoomIn} className="btn">
                <LuZoomIn />
              </button>
              <button onClick={resetZoom}>Reset zoom</button>
            </div>
            {
              <VictoryChart
                theme={VictoryTheme.material}
                domain={initialZoomDomain}
                containerComponent={
                  <VictoryZoomContainer
                    zoomDomain={zoomDomain || initialZoomDomain}
                    onZoomDomainChange={handleZoom}
                  />
                }
                height={300}
                width={300}
              >
                <VictoryAxis
                  style={{
                    axis: { stroke: 'transparent' },
                    ticks: { stroke: 'transparent' },
                    tickLabels: { fill: 'transparent' },
                  }}
                />
                <VictoryScatter
                  style={{
                    data: {
                      fill: ({ datum }) =>
                        datum.index === currentElementId
                          ? 'black'
                          : labelColorMapping[datum.labels],
                      opacity: ({ datum }) => (datum.index === currentElementId ? 1 : 0.5),
                      cursor: 'pointer',
                    },
                  }}
                  size={({ datum }) => (datum.index === currentElementId ? 5 : 2)}
                  labels={({ datum }) => datum.index}
                  labelComponent={
                    <VictoryTooltip style={{ fontSize: 10 }} flyoutStyle={{ fill: 'white' }} />
                  }
                  data={projectionData.x.map((value, index) => {
                    return {
                      x: value,
                      y: projectionData.y[index],
                      labels: projectionData.labels[index],
                      //texts: projectionData.texts[index],
                      index: projectionData.index[index],
                    };
                  })}
                  events={[
                    {
                      target: 'data',
                      eventHandlers: {
                        onClick: (_, props) => {
                          const { datum } = props;
                          getElementById(datum.index, 'train').then((element) => {
                            setSelectedElement(element || null);
                          });
                          //navigate(`/projects/${projectName}/annotate/${datum.index}`);
                        },
                      },
                    },
                  ]}
                />

                <VictoryLegend
                  x={0}
                  y={0}
                  title="Legend"
                  centerTitle
                  orientation="vertical"
                  gutter={10}
                  style={{
                    border: { stroke: 'black' },
                    title: { fontSize: 5 },
                    labels: { fontSize: 5 },
                  }}
                  data={Object.keys(labelColorMapping).map((label) => ({
                    name: label,
                    symbol: { fill: labelColorMapping[label] },
                  }))}
                />
              </VictoryChart>
            }
          </div>
          <div className="col-4">
            {selectedElement && (
              <div className="mt-5">
                Element:{' '}
                <div className="badge bg-light text-dark">{selectedElement.element_id}</div>
                <div className="mt-2">{selectedElement.text}</div>
                <div className="mt-2">
                  Previous annotations : {JSON.stringify(selectedElement.history)}
                </div>
                <button
                  className="btn btn-primary mt-3"
                  onClick={() =>
                    navigate(`/projects/${projectName}/annotate/${selectedElement.element_id}`)
                  }
                >
                  Annotate
                </button>
              </div>
            )}
          </div>
        </div>
      )}
      <form onSubmit={handleSubmit(onSubmit)}>
        <label htmlFor="model">Select a model</label>
        <select id="model" {...register('method')}>
          <option value=""></option>
          {Object.keys(availableProjections ? availableProjections : []).map((e) => (
            <option key={e} value={e}>
              {e}
            </option>
          ))}{' '}
        </select>
        <div>
          <label htmlFor="features">Select features</label>
          <Controller
            name="features"
            control={control}
            render={({ field: { value, onChange } }) => (
              <Select
                options={features}
                isMulti
                value={features.filter((feature) => value?.includes(feature.value))}
                onChange={(selectedOptions) => {
                  onChange(selectedOptions ? selectedOptions.map((option) => option.value) : []);
                }}
              />
            )}
          />
        </div>
        {availableProjections && selectedMethod == 'tsne' && (
          <div>
            <label htmlFor="perplexity">perplexity</label>
            <input
              type="number"
              step="1"
              id="perplexity"
              {...register('params.perplexity', { valueAsNumber: true })}
            ></input>
            <label>Learning rate</label>
            <select {...register('params.learning_rate')}>
              <option key="auto" value="auto">
                auto
              </option>
            </select>
            <label>Init</label>
            <select {...register('params.init')}>
              <option key="random" value="random">
                random
              </option>
            </select>
          </div>
        )}
        {availableProjections && selectedMethod == 'umap' && (
          <div>
            <label htmlFor="n_neighbors">n_neighbors</label>
            <input
              type="number"
              step="1"
              id="n_neighbors"
              {...register('params.n_neighbors', { valueAsNumber: true })}
            ></input>
            <label htmlFor="min_dist">min_dist</label>
            <input
              type="number"
              id="min_dist"
              step="0.01"
              {...register('params.min_dist', { valueAsNumber: true })}
            ></input>
            <label htmlFor="metric">Metric</label>
            <select {...register('params.metric')}>
              <option key="cosine" value="cosine">
                cosine
              </option>
              <option key="euclidean" value="euclidean">
                euclidean
              </option>
            </select>
          </div>
        )}
        <label htmlFor="n_components">n_components</label>
        <input
          type="number"
          id="n_components"
          step="1"
          {...register('params.n_components', { valueAsNumber: true, required: true })}
        ></input>

        <button className="btn btn-primary btn-validation">Compute</button>
      </form>
    </div>
  );
};
