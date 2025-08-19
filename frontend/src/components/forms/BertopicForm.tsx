import { FC } from 'react';
import { SubmitHandler, useForm } from 'react-hook-form';
import { useComputeBertopic } from '../../core/api';
import { ComputeBertopicModel } from '../../types';
interface BertopicCreationFormProps {
  projectSlug: string | null;
  availableModels: string[];
}

export const BertopicForm: FC<BertopicCreationFormProps> = ({ projectSlug, availableModels }) => {
  const { computeBertopic } = useComputeBertopic(projectSlug);
  const { handleSubmit: handleSubmitNewModel, register } = useForm<ComputeBertopicModel>({
    defaultValues: {
      name: 'bertopic',
      outlier_reduction: true,
      min_topic_size: 10,
      nr_topics: 'auto',
      hdbscan_min_cluster_size: 10,
      umap_n_neighbors: 10,
      umap_n_components: 2,
      umap_min_dist: 0.0,
      embedding_model: availableModels[0],
      force_compute_embeddings: false,
      filter_text_length: 2,
    },
  });

  const onSubmitNewModel: SubmitHandler<ComputeBertopicModel> = async (data) => {
    await computeBertopic(data);
  };
  return (
    <form onSubmit={handleSubmitNewModel(onSubmitNewModel)}>
      <div className="d-flex items-center gap-2">
        <label className="form-label" htmlFor="name">
          Name
        </label>
        <input className="form-control" id="name" type="text" {...register('name')} />
      </div>

      <details className="custom-details">
        <summary>Advanced parameters</summary>
        <label className="form-label" htmlFor="outlier_reduction">
          Outlier reduction
          <input
            id="outlier_reduction"
            type="checkbox"
            {...register('outlier_reduction')}
            className="mx-2"
          />
        </label>
        <label className="form-label" htmlFor="force_compute_embeddings">
          Force compute embeddings
          <input
            id="force_compute_embeddings"
            type="checkbox"
            {...register('force_compute_embeddings')}
            className="mx-2"
          />
        </label>
        <label className="form-label" htmlFor="filter_text_length">
          Filter text length lower than
          <input
            className="form-control"
            id="filter_text_length"
            type="number"
            {...register('filter_text_length')}
          />
        </label>
        <label className="form-label" htmlFor="embedding_model">
          Embedding model
          <select className="form-select" {...register('embedding_model')}>
            {availableModels.map((model) => (
              <option key={model} value={model}>
                {model}
              </option>
            ))}
          </select>
        </label>
        <label className="form-label" htmlFor="min_topic_size">
          Min topic size
          <input
            className="form-control"
            id="minTopicSize"
            type="number"
            {...register('min_topic_size')}
          />
        </label>
        <label className="form-label" htmlFor="nr_topics">
          Number of topics (auto if null)
          <input className="form-control" id="nr_topics" type="number" {...register('nr_topics')} />
        </label>
        <label className="form-label" htmlFor="hdbscan_min_cluster_size">
          HDBSCAN min cluster size
          <input
            className="form-control"
            id="hdbscan_min_cluster_size"
            type="number"
            {...register('hdbscan_min_cluster_size')}
          />
        </label>
        <label className="form-label" htmlFor="umap_n_neighbors">
          UMAP n_neighbors
          <input
            className="form-control"
            id="umap_n_neighbors"
            type="number"
            {...register('umap_n_neighbors')}
          />
        </label>
        <label className="form-label" htmlFor="umap_n_components">
          UMAP umap_n_components
          <input
            className="form-control"
            id="umap_n_components"
            type="number"
            {...register('umap_n_components')}
          />
        </label>
        <label className="form-label" htmlFor="umap_min_dist">
          UMAP umap_min_dist
          <input
            className="form-control"
            id="umap_min_dist"
            type="number"
            step="any"
            {...register('umap_min_dist')}
          />
        </label>
      </details>

      <button className="btn btn-primary w-50">Compute Bertopic on trainset</button>
    </form>
  );
};
